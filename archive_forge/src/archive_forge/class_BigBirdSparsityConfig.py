import random
import torch
class BigBirdSparsityConfig(SparsityConfig):
    """Configuration class to store `BigBird` sparsity configuration.
    For more details about this sparsity config, please see `Big Bird: Transformers for
    Longer Sequences`: https://arxiv.org/pdf/2007.14062.pdf
    This class extends parent class of `SparsityConfig` and customizes it for `BigBird` sparsity.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False, num_random_blocks=1, num_sliding_window_blocks=3, num_global_blocks=1, attention='bidirectional'):
        """Initialize the BigBird Sparsity Pattern Config.
        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of
                sparse self-attention is based on blocked sparse matrices. In which this parameter
                defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned
                a different sparsity layout; default is false and this will be satisfied based on
                availability.
             num_random_blocks: optional: an integer determining the number of random blocks in each
                block row.
             num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding
                local attention window.
             num_global_blocks: optional: an integer determining how many consecutive blocks, starting
                from index 0, are considered as global attention. Global block tokens will be attended
                by all other block tokens and will attend to all other block tokens as well.
             attention: optional: a string determining attention type. Attention can be `unidirectional`,
                such as autoregressive models, in which tokens attend only to tokens appear before them
                in the context. Considering that, the upper triangular of attention matrix is empty as
                above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to
                any other tokens before or after them. Then, the upper triangular part of the attention
                matrix is mirror of the lower triangular in the above figure.
        """
        super().__init__(num_heads, block_size, different_layout_per_head)
        self.num_random_blocks = num_random_blocks
        self.num_sliding_window_blocks = num_sliding_window_blocks
        self.num_global_blocks = num_global_blocks
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only "uni/bi-directional" attentions are supported for now!')
        self.attention = attention

    def set_random_layout(self, h, layout):
        """Sets random attention layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless
        `different_layout_per_head` parameter is set in which each head can have a different random layout.
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
            sample_range = range(0, num_blocks) if self.attention == 'bidirectional' else range(0, row + 1)
            rnd_cols = random.sample(sample_range, self.num_random_blocks)
            layout[h, row, rnd_cols] = 1
        return layout

    def set_sliding_window_layout(self, h, layout):
        """Sets sliding local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which local sliding window layout is set
        """
        num_blocks = layout.shape[1]
        if num_blocks < self.num_sliding_window_blocks:
            raise ValueError(f'Number of sliding window blocks, {self.num_sliding_window_blocks}, must be smaller than\n                overall number of blocks in a row, {num_blocks}!')
        w = self.num_sliding_window_blocks // 2
        for row in range(0, num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, num_blocks)
            layout[h, row, start:end] = 1
        return layout

    def set_global_layout_itc(self, h, layout):
        """Sets global attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout
                of all head in which global layout is set
        """
        num_blocks = layout.shape[1]
        if num_blocks < self.num_global_blocks:
            raise ValueError(f'Number of global blocks, {self.num_global_blocks}, must be smaller than overall number\n                of blocks in a row, {num_blocks}!')
        layout[h, 0:self.num_global_blocks, :] = 1
        layout[h, :, 0:self.num_global_blocks] = 1
        if self.attention == 'unidirectional':
            layout = torch.tril(layout)
        return layout

    def make_layout(self, seq_len):
        """Generates `BigBird` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `BigBird`
             sparsity layout of all head
        """
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_random_layout(h, layout)
            layout = self.set_sliding_window_layout(h, layout)
            layout = self.set_global_layout_itc(h, layout)
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout