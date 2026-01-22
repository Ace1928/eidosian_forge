from typing import Optional, Tuple
import torch
class ConvTasNet(torch.nn.Module):
    """Conv-TasNet architecture introduced in
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    :cite:`Luo_2019`.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
    """

    def __init__(self, num_sources: int=2, enc_kernel_size: int=16, enc_num_feats: int=512, msk_kernel_size: int=3, msk_num_feats: int=128, msk_num_hidden_feats: int=512, msk_num_layers: int=8, msk_num_stacks: int=3, msk_activate: str='sigmoid'):
        super().__init__()
        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.encoder = torch.nn.Conv1d(in_channels=1, out_channels=enc_num_feats, kernel_size=enc_kernel_size, stride=self.enc_stride, padding=self.enc_stride, bias=False)
        self.mask_generator = MaskGenerator(input_dim=enc_num_feats, num_sources=num_sources, kernel_size=msk_kernel_size, num_feats=msk_num_feats, num_hidden=msk_num_hidden_feats, num_layers=msk_num_layers, num_stacks=msk_num_stacks, msk_activate=msk_activate)
        self.decoder = torch.nn.ConvTranspose1d(in_channels=enc_num_feats, out_channels=1, kernel_size=enc_kernel_size, stride=self.enc_stride, padding=self.enc_stride, bias=False)

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return (input, 0)
        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(batch_size, num_channels, num_paddings, dtype=input.dtype, device=input.device)
        return (torch.cat([input, pad], 2), num_paddings)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f'Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}')
        padded, num_pads = self._align_num_frames_with_strides(input)
        batch_size, num_padded_frames = (padded.shape[0], padded.shape[2])
        feats = self.encoder(padded)
        masked = self.mask_generator(feats) * feats.unsqueeze(1)
        masked = masked.view(batch_size * self.num_sources, self.enc_num_feats, -1)
        decoded = self.decoder(masked)
        output = decoded.view(batch_size, self.num_sources, num_padded_frames)
        if num_pads > 0:
            output = output[..., :-num_pads]
        return output