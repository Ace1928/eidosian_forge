import torch

        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
        Returns:
            Tensor: Predictor tensor of dimension (batch, time, class).
        