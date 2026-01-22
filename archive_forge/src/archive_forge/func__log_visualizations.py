import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _log_visualizations(self, session_df: Any) -> None:
    if not (self.visualizations and self.nlp):
        return
    spacy = import_spacy()
    prompts = session_df['prompts'].tolist()
    outputs = session_df['text'].tolist()
    for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
        doc = self.nlp(output)
        sentence_spans = list(doc.sents)
        for visualization in self.visualizations:
            try:
                html = spacy.displacy.render(sentence_spans, style=visualization, options={'compact': True}, jupyter=False, page=True)
                self.experiment.log_asset_data(html, name=f'langchain-viz-{visualization}-{idx}.html', metadata={'prompt': prompt}, step=idx)
            except Exception as e:
                self.comet_ml.LOGGER.warning(e, exc_info=True, extra={'show_traceback': True})
    return