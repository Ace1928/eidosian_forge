import warnings
def _get_output_shape_given_tf_padding(input_size, kernel_size, strides, padding, output_padding, dilation_rate):
    if input_size is None:
        return None
    assert padding.lower() in {'valid', 'same'}
    kernel_size = (kernel_size - 1) * dilation_rate + 1
    if padding.lower() == 'valid':
        output_padding = max(kernel_size, strides) - kernel_size if output_padding is None else output_padding
        return (input_size - 1) * strides + kernel_size + output_padding
    elif output_padding is None:
        return input_size * strides
    else:
        return (input_size - 1) * strides + kernel_size % 2 + output_padding