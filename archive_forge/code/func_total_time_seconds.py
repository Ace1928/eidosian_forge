import torch._C
@property
def total_time_seconds(self):
    return self.num_iters * (self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads