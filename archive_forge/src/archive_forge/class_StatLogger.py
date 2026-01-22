from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
class StatLogger:
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float, labels: Dict[str, str]) -> None:
        self.last_local_log = time.monotonic()
        self.local_interval = local_interval
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()))

    def info(self, type: str, obj: object) -> None:
        if type == 'cache_config':
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        self.metrics.gauge_scheduler_running.labels(**self.labels).set(stats.num_running)
        self.metrics.gauge_scheduler_swapped.labels(**self.labels).set(stats.num_swapped)
        self.metrics.gauge_scheduler_waiting.labels(**self.labels).set(stats.num_waiting)
        self.metrics.gauge_gpu_cache_usage.labels(**self.labels).set(stats.gpu_cache_usage)
        self.metrics.gauge_cpu_cache_usage.labels(**self.labels).set(stats.cpu_cache_usage)
        self.metrics.counter_prompt_tokens.labels(**self.labels).inc(stats.num_prompt_tokens)
        self.metrics.counter_generation_tokens.labels(**self.labels).inc(stats.num_generation_tokens)
        for ttft in stats.time_to_first_tokens:
            self.metrics.histogram_time_to_first_token.labels(**self.labels).observe(ttft)
        for tpot in stats.time_per_output_tokens:
            self.metrics.histogram_time_per_output_token.labels(**self.labels).observe(tpot)
        for e2e in stats.time_e2e_requests:
            self.metrics.histogram_e2e_request_latency.labels(**self.labels).observe(e2e)

    def _log_prometheus_interval(self, prompt_throughput: float, generation_throughput: float) -> None:
        self.metrics.gauge_avg_prompt_throughput.labels(**self.labels).set(prompt_throughput)
        self.metrics.gauge_avg_generation_throughput.labels(**self.labels).set(generation_throughput)

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to prometheus and tracked stats every iteration. 
           Logs to Stdout every self.local_interval seconds."""
        self._log_prometheus(stats)
        self.num_prompt_tokens.append(stats.num_prompt_tokens)
        self.num_generation_tokens.append(stats.num_generation_tokens)
        if self._local_interval_elapsed(stats.now):
            prompt_throughput = self._get_throughput(self.num_prompt_tokens, now=stats.now)
            generation_throughput = self._get_throughput(self.num_generation_tokens, now=stats.now)
            self._log_prometheus_interval(prompt_throughput=prompt_throughput, generation_throughput=generation_throughput)
            logger.info(f'Avg prompt throughput: {prompt_throughput:.1f} tokens/s, Avg generation throughput: {generation_throughput:.1f} tokens/s, Running: {stats.num_running} reqs, Swapped: {stats.num_swapped} reqs, Pending: {stats.num_waiting} reqs, GPU KV cache usage: {stats.gpu_cache_usage * 100:.1f}%, CPU KV cache usage: {stats.cpu_cache_usage * 100:.1f}%')
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now