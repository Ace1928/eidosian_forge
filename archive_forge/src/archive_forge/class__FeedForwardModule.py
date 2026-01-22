from typing import Optional, Tuple
import torch
class _FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float=0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, hidden_dim, bias=True), torch.nn.SiLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, input_dim, bias=True), torch.nn.Dropout(dropout))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)