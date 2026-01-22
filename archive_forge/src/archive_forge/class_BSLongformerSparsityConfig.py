import random
import torch
class BSLongformerSparsityConfig(SparsityConfig):
    """Configuration class to store edited `Longformer` sparsity configuration.
    Note) this is a block-sparse version of the Longformer which is slightly different than original
    Longformer; which is element-wise sparsity.
    For more details about this sparsity config, please see `Longformer:
    The Long-Document Transformer`: https://arxiv.org/pdf/2004.05150.pdf
    This class extends parent class of `SparsityConfig` and customizes it for `Longformer` sparsity.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False, num_sliding_window_blocks=3, global_block_indices=[0], global_block_end_indices=None, attention='bidirectional'):
        """Initialize the edited `Longformer` Sparsity Pattern Config.
        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of sparse
                self-attention is based on blocked sparse matrices. In which this parameter defines size
                of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a
                different sparsity layout; default is false and this will be satisfied based on
                availability.
             num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding
                local attention window.
             global_block_indices: optional: a list of integers determining which blocks are considered
                as global attention. Given indices, determine the blocks that all other token blocks
                attend to and they attend to all other token blocks. Default value is only index 0.
                Notice that if global_block_end_indices parameter is set, this parameter is used as
                starting index of each global window.
             global_block_end_indices: optional: a list of integers determining end indices of global
                window blocks. By default this is not used. But if it is set, it must have the same size
                of global_block_indices parameter, and combining this two parameters, for each index i,
                blocks from global_block_indices[i] to global_block_end_indices[i] (exclusive) are
                considered as global attention.
             attention: optional: a string determining attention type. Attention can be `unidirectional`,
                such as autoregressive models, in which tokens attend only to tokens appear before them
                in the context. Considering that, the upper triangular of attention matrix is empty as
                above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to
                any other tokens before or after them. Then, the upper triangular part of the attention
                matrix is mirror of the lower triangular in the above figure.
        """
        super().__init__(num_heads, block_size, different_layout_per_head)
        self.num_sliding_window_blocks = num_sliding_window_blocks
        self.global_block_indices = global_block_indices
        self.attention = attention
        if global_block_end_indices is not None:
            if len(global_block_indices) != len(global_block_end_indices):
                raise ValueError(f'Global block start indices length, {len(global_block_indices)}, must be same as\n                    global block end indices length, {len(global_block_end_indices)}!')
            for _, (start_idx, end_idx) in enumerate(zip(global_block_indices, global_block_end_indices)):
                if start_idx >= end_idx:
                    raise ValueError(f'Global block start index, {start_idx}, must be smaller than global block end\n                        index, {end_idx}!')
        self.global_block_end_indices = global_block_end_indices

    def set_sliding_window_layout(self, h, layout):
        """Sets sliding local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout
                of all head in which local sliding window layout is set
        """
        num_blocks = layout.shape[1]
        if num_blocks < self.num_sliding_window_blocks:
            raise ValueError(f'Number of sliding window blocks, {self.num_sliding_window_blocks}, must be smaller\n                than overall number of blocks in a row, {num_blocks}!')
        w = self.num_sliding_window_blocks // 2
        for row in range(0, num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, num_blocks)
            layout[h, row, start:end] = 1
        return layout

    def set_global_layout(self, h, layout):
        """Sets global attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which global layout is set
        """
        num_blocks = layout.shape[1]
        if self.global_block_end_indices is None:
            for idx in self.global_block_indices:
                if idx < num_blocks:
                    layout[h, idx, :] = 1
                    layout[h, :, idx] = 1
        else:
            for _, (start_idx, end_idx) in enumerate(zip(self.global_block_indices, self.global_block_end_indices)):
                if start_idx < num_blocks:
                    end_idx = min(end_idx, num_blocks)
                    layout[h, start_idx:end_idx, :] = 1
                    layout[h, :, start_idx:end_idx] = 1
        if self.attention == 'unidirectional':
            layout = torch.tril(layout)
        return layout

    def make_layout(self, seq_len):
        """Generates edited `Longformer` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `BSLongformer`
                sparsity layout of all head
        """
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_sliding_window_layout(h, layout)
            layout = self.set_global_layout(h, layout)
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout