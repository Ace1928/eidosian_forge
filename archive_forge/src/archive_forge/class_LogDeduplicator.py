import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import (
from ray.util.debug import log_once
class LogDeduplicator:

    def __init__(self, agg_window_s: int, allow_re: Optional[str], skip_re: Optional[str], *, _timesource=None):
        self.agg_window_s = agg_window_s
        if allow_re:
            self.allow_re = re.compile(allow_re)
        else:
            self.allow_re = None
        if skip_re:
            self.skip_re = re.compile(skip_re)
        else:
            self.skip_re = None
        self.recent: Dict[str, DedupState] = {}
        self.timesource = _timesource or (lambda: time.time())
        run_callback_on_events_in_ipython('post_execute', self.flush)

    def deduplicate(self, batch: LogBatch) -> List[LogBatch]:
        """Rewrite a batch of lines to reduce duplicate log messages.

        Args:
            batch: The batch of lines from a single source.

        Returns:
            List of batches from this and possibly other previous sources to print.
        """
        if not RAY_DEDUP_LOGS:
            return [batch]
        now = self.timesource()
        metadata = batch.copy()
        del metadata['lines']
        source = (metadata.get('ip'), metadata.get('pid'))
        output: List[LogBatch] = [dict(**metadata, lines=[])]
        for line in batch['lines']:
            if RAY_TQDM_MAGIC in line or (self.allow_re and self.allow_re.search(line)):
                output[0]['lines'].append(line)
                continue
            elif self.skip_re and self.skip_re.search(line):
                continue
            dedup_key = _canonicalise_log_line(line)
            if dedup_key in self.recent:
                sources = self.recent[dedup_key].sources
                sources.add(source)
                if len(sources) > 1 or batch['pid'] == 'raylet':
                    state = self.recent[dedup_key]
                    self.recent[dedup_key] = DedupState(state.timestamp, state.count + 1, line, metadata, sources)
                else:
                    output[0]['lines'].append(line)
            else:
                self.recent[dedup_key] = DedupState(now, 0, line, metadata, {source})
                output[0]['lines'].append(line)
        while self.recent:
            if now - next(iter(self.recent.values())).timestamp < self.agg_window_s:
                break
            dedup_key = next(iter(self.recent))
            state = self.recent.pop(dedup_key)
            if state.count > 1:
                output.append(dict(**state.metadata, lines=[state.formatted()]))
                state.timestamp = now
                state.count = 0
                self.recent[dedup_key] = state
            elif state.count > 0:
                output.append(dict(state.metadata, lines=[state.line]))
        return output

    def flush(self) -> List[dict]:
        """Return all buffered log messages and clear the buffer.

        Returns:
            List of log batches to print.
        """
        output = []
        for state in self.recent.values():
            if state.count > 1:
                output.append(dict(state.metadata, lines=[state.formatted()]))
            elif state.count > 0:
                output.append(dict(state.metadata, **{'lines': [state.line]}))
        self.recent.clear()
        return output