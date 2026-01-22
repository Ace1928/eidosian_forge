import warnings
Convert the padding arguments from Keras to the ones used by Torch.
    Torch starts with an output shape of `(input-1) * stride + kernel_size`,
    then removes `torch_padding` from both sides, and adds
    `torch_output_padding` on the right.
    Because in Torch the output_padding can only be added to the right,
    consistency with Tensorflow is not always possible. In particular this is
    the case when both the Torch padding and output_padding values are stricly
    positive.
    