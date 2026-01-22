import torch
from torch.ao.quantization.observer import ObserverBase
def _calculate_range_stats(self, x_copy):
    """Calculates and stores range stats with forward values.

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    min_val_cur, max_val_cur = torch.aminmax(x_copy)
    epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
    epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)
    self.epoch_activation_min.copy_(epoch_min_val)
    self.epoch_activation_max.copy_(epoch_max_val)
    current_batch_range = max_val_cur - min_val_cur
    new_range = (self.average_batch_activation_range * self.num_batches_tracked + current_batch_range) / (self.num_batches_tracked + 1)
    self.average_batch_activation_range = new_range
    self.num_batches_tracked += 1
    return x_copy