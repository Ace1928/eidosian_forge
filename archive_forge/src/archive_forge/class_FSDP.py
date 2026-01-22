import torch.nn as nn
from fairscale.optim import GradScaler
class FSDP:

    def get_model_config():
        return {'vocab_size': 10000, 'ninp': 2048, 'nhid': 2048, 'nhead': 32, 'dropout': 0, 'initrange': 0.1, 'scaler': GradScaler(), 'clip_value': 0.05, 'num_decoder_layers': 10, 'seq_len': 32}

    def get_benchmark_config():
        return {'epochs': 1, 'lr': 0.001, 'batch_size': 8, 'criterion': nn.CrossEntropyLoss()}

    def get_golden_real_stats():
        raise NotImplementedError('Synthetic data benchmarks are not supported.')

    def get_golden_synthetic_stats():
        return {'avg_wps': 486.303, 'std_dev_wps': 71.307, 'peak_mem_usage': [5.5055 * 2 ** 30, 5.5055 * 2 ** 30, 5.5055 * 2 ** 30, 5.5055 * 2 ** 30]}