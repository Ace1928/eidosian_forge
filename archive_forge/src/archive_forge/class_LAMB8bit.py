from bitsandbytes.optim.optimizer import Optimizer2State
class LAMB8bit(Optimizer2State):

    def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, adam_w_mode=True, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=False, max_unorm=1.0):
        """
        8-bit LAMB optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            bias_correction (`bool`, defaults to `True`):
                Whether to apply bias correction to the first and second-order moments.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
            adam_w_mode (`bool`, defaults to `True`):
                Whether to use the AdamW variant.
            args (`dict`, defaults to `None`):
                A dictionary with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 1.0):
                The maximum gradient norm.
        """
        super().__init__('lamb', params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, max_unorm=1.0)