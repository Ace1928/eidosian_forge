import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
class WaveRNN(nn.Module):
    """WaveRNN model from *Efficient Neural Audio Synthesis* :cite:`wavernn`
    based on the implementation from `fatchord/WaveRNN <https://github.com/fatchord/WaveRNN>`_.

    The original implementation was introduced in *Efficient Neural Audio Synthesis*
    :cite:`kalchbrenner2018efficient`. The input channels of waveform and spectrogram have to be 1.
    The product of `upsample_scales` must equal `hop_length`.

    See Also:
        * `Training example <https://github.com/pytorch/audio/tree/release/0.12/examples/pipeline_wavernn>`__
        * :class:`torchaudio.pipelines.Tacotron2TTSBundle`: TTS pipeline with pretrained model.

    Args:
        upsample_scales: the list of upsample scales.
        n_classes: the number of output classes.
        hop_length: the number of samples between the starts of consecutive frames.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_rnn: the dimension of RNN layer. (Default: ``512``)
        n_fc: the dimension of fully connected layer. (Default: ``512``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)

    Example
        >>> wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
        >>> waveform, sample_rate = torchaudio.load(file)
        >>> # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
        >>> specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
        >>> output = wavernn(waveform, specgram)
        >>> # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
    """

    def __init__(self, upsample_scales: List[int], n_classes: int, hop_length: int, n_res_block: int=10, n_rnn: int=512, n_fc: int=512, kernel_size: int=5, n_freq: int=128, n_hidden: int=128, n_output: int=128) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self._pad = (kernel_size - 1 if kernel_size % 2 else kernel_size) // 2
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.n_bits: int = int(math.log2(self.n_classes))
        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != self.hop_length:
            raise ValueError(f'Expected: total_scale == hop_length, but found {total_scale} != {hop_length}')
        self.upsample = UpsampleNetwork(upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)
        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, self.n_classes)

    def forward(self, waveform: Tensor, specgram: Tensor) -> Tensor:
        """Pass the input through the WaveRNN model.

        Args:
            waveform: the input waveform to the WaveRNN layer (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
            specgram: the input spectrogram to the WaveRNN layer (n_batch, 1, n_freq, n_time)

        Return:
            Tensor: shape (n_batch, 1, (n_time - kernel_size + 1) * hop_length, n_classes)
        """
        if waveform.size(1) != 1:
            raise ValueError('Require the input channel of waveform is 1')
        if specgram.size(1) != 1:
            raise ValueError('Require the input channel of specgram is 1')
        waveform, specgram = (waveform.squeeze(1), specgram.squeeze(1))
        batch_size = waveform.size(0)
        h1 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        h2 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        specgram, aux = self.upsample(specgram)
        specgram = specgram.transpose(1, 2)
        aux = aux.transpose(1, 2)
        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        x = torch.cat([waveform.unsqueeze(-1), specgram, a1], dim=-1)
        x = self.fc(x)
        res = x
        x, _ = self.rnn1(x, h1)
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=-1)
        x, _ = self.rnn2(x, h2)
        x = x + res
        x = torch.cat([x, a3], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = torch.cat([x, a4], dim=-1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x.unsqueeze(1)

    @torch.jit.export
    def infer(self, specgram: Tensor, lengths: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Inference method of WaveRNN.

        This function currently only supports multinomial sampling, which assumes the
        network is trained on cross entropy loss.

        Args:
            specgram (Tensor):
                Batch of spectrograms. Shape: `(n_batch, n_freq, n_time)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``specgram`` contains spectrograms with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The inferred waveform of size `(n_batch, 1, n_time)`.
                1 stands for a single channel.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        device = specgram.device
        dtype = specgram.dtype
        specgram = torch.nn.functional.pad(specgram, (self._pad, self._pad))
        specgram, aux = self.upsample(specgram)
        if lengths is not None:
            lengths = lengths * self.upsample.total_scale
        output: List[Tensor] = []
        b_size, _, seq_len = specgram.size()
        h1 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        h2 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        x = torch.zeros((b_size, 1), device=device, dtype=dtype)
        aux_split = [aux[:, self.n_aux * i:self.n_aux * (i + 1), :] for i in range(4)]
        for i in range(seq_len):
            m_t = specgram[:, :, i]
            a1_t, a2_t, a3_t, a4_t = [a[:, :, i] for a in aux_split]
            x = torch.cat([x, m_t, a1_t], dim=1)
            x = self.fc(x)
            _, h1 = self.rnn1(x.unsqueeze(1), h1)
            x = x + h1[0]
            inp = torch.cat([x, a2_t], dim=1)
            _, h2 = self.rnn2(inp.unsqueeze(1), h2)
            x = x + h2[0]
            x = torch.cat([x, a3_t], dim=1)
            x = F.relu(self.fc1(x))
            x = torch.cat([x, a4_t], dim=1)
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            posterior = F.softmax(logits, dim=1)
            x = torch.multinomial(posterior, 1).float()
            x = 2 * x / (2 ** self.n_bits - 1.0) - 1.0
            output.append(x)
        return (torch.stack(output).permute(1, 2, 0), lengths)