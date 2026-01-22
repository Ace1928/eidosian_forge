import torch.nn as nn
from fairscale.optim import GradScaler
class Offload_Transformer:

    def get_model_config():
        return {'vocab_size': 10000, 'ninp': 2048, 'nhid': 2048, 'nhead': 32, 'dropout': 0, 'initrange': 0.1, 'scaler': GradScaler(), 'clip_value': 0.05, 'num_decoder_layers': 10, 'seq_len': 32}

    def get_benchmark_config(checkpoint_activation=True):
        return {'epochs': 1, 'lr': 0.001, 'batch_size': 8, 'criterion': nn.CrossEntropyLoss(), 'checkpoint_activation': checkpoint_activation, 'num_microbatches': 1, 'slices': 3}

    def get_golden_real_stats():
        return {'avg_wps': 192.105, 'std_dev_wps': 39.56, 'peak_mem_usage': 1180848128}