import random
import torch
class VariableSparsityConfig(SparsityConfig):
    """Configuration class to store `Variable` sparsity configuration.
    This layout is an extension of FixedSparsityConfig in which:
      - user can set random layout; default value is zero means no random block
      - user can provide a list of local block sizes
      - user can provide a list of global block indices.
    For more details about `Fixed` sparsity config, please see `Generative Modeling with
    Sparse Transformers`: https://arxiv.org/abs/1904.10509; this has been customized.
    This class extends parent class of `SparsityConfig` and customizes it for `Fixed` sparsity.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False, num_random_blocks=0, local_window_blocks=[4], global_block_indices=[0], global_block_end_indices=None, attention='bidirectional', horizontal_global_attention=False):
        """Initialize `Variable` Sparsity Pattern Config.
        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of sparse
                self-attention is based on blocked sparse matrices. In which this parameter defines
                size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a
                different sparsity layout; default is false and this will be satisfied based on
                availability. Currently this sparsity config can only assign single layout to all heads;
                needs to be extended for different layout per head.
             num_random_blocks: optional: an integer determining the number of random blocks in each block row.
             local_window_blocks: optional: a list of integers determining the number of blocks in each
                local attention window. It assumes first number determines # of blocks in the first local
                window, second the second window, ..., and the last number determines the number of blocks
                in the remaining local windows.
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
             horizontal_global_attention: optional: a boolean determining if blocks that are global
                representative of a local window, also attend to all other blocks. This is valid only if
                attention type is `bidirectional`. Looking at the attention matrix, that means global
                attention not only includes the vertical blocks, but also horizontal blocks.
        """
        super().__init__(num_heads, block_size, different_layout_per_head)
        self.num_random_blocks = num_random_blocks
        self.local_window_blocks = local_window_blocks
        self.global_block_indices = global_block_indices
        if global_block_end_indices is not None:
            if len(global_block_indices) != len(global_block_end_indices):
                raise ValueError(f'Global block start indices length, {len(global_block_indices)}, must be same as\n                    global block end indices length, {len(global_block_end_indices)}!')
            for _, (start_idx, end_idx) in enumerate(zip(global_block_indices, global_block_end_indices)):
                if start_idx >= end_idx:
                    raise ValueError(f'Global block start index, {start_idx}, must be smaller than global block end\n                        index, {end_idx}!')
        self.global_block_end_indices = global_block_end_indices
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only "uni/bi-directional" attentions are supported for now!')
        self.attention = attention
        if attention != 'bidirectional' and horizontal_global_attention:
            raise ValueError('only "bi-directional" attentions can support horizontal global attention!')
        self.horizontal_global_attention = horizontal_global_attention

    def set_random_layout(self, h, layout):
        """Sets random attention layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless
        `different_layout_per_head` parameter is set in which each head can have a different random
        layout.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which random layout is set
        """
        num_blocks = layout.shape[1]
        if num_blocks < self.num_random_blocks:
            raise ValueError(f'Number of random blocks, {self.num_random_blocks}, must be smaller than overall number\n                of blocks in a row, {num_blocks}!')
        for row in range(0, num_blocks):
            rnd_cols = random.sample(range(0, num_blocks), self.num_random_blocks)
            layout[h, row, rnd_cols] = 1
        return layout

    def set_local_layout(self, h, layout):
        """Sets local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which local layout is set
        """
        num_blocks = layout.shape[1]
        start_block_idx = 0
        end_block_idx = 0
        for block_size in self.local_window_blocks:
            end_block_idx += block_size
            end_block_idx = min(end_block_idx, num_blocks)
            for row in range(start_block_idx, end_block_idx):
                for col in range(start_block_idx, row + 1 if self.attention == 'unidirectional' else end_block_idx):
                    layout[h, row, col] = 1
            start_block_idx += block_size
        for i in range(start_block_idx, num_blocks, block_size):
            end_block_idx = min(i + block_size, num_blocks)
            for row in range(i, end_block_idx):
                for col in range(i, row + 1 if self.attention == 'unidirectional' else end_block_idx):
                    layout[h, row, col] = 1
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
                    if self.horizontal_global_attention:
                        layout[h, idx, :] = 1
                    first_row = 0 if self.attention == 'bidirectional' else idx
                    layout[h, first_row:, idx] = 1
        else:
            for _, (start_idx, end_idx) in enumerate(zip(self.global_block_indices, self.global_block_end_indices)):
                if start_idx < num_blocks:
                    end_idx = min(end_idx, num_blocks)
                    if self.horizontal_global_attention:
                        layout[h, start_idx:end_idx, :] = 1
                    first_row = 0 if self.attention == 'bidirectional' else start_idx
                    layout[h, first_row:, start_idx:end_idx] = 1
        return layout

    def make_layout(self, seq_len):
        """Generates `Variable` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `Variable`
                sparsity layout of all head
        """
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_random_layout(h, layout)
            layout = self.set_local_layout(h, layout)
            layout = self.set_global_layout(h, layout)
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout