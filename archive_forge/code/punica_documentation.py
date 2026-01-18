from typing import Optional
import torch

    Same as `add_lora` but you can operate on slices of y.
    Pass whole y, define y_offset and y_slice_size.

    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_t_all: Shape: `[None, L, R, H1]`. All of the transposed
        LoRA A matrices.
      wb_t_all: Shape: `[None, L, H2, R]`. All of the transposed
        LoRA B matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
      y_offset: Offset to apply to the starting column of y.
      y_slice_size: Size of the y column slice.
    