from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
Creates an object that stores options for a Checkpoint.

    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.

        This is for example useful if you want to save to a local directory,
        such as "/tmp" when running in a distributed setting. In that case pass
        a device for the host where the "/tmp" directory is accessible.

      experimental_enable_async_checkpoint: bool Type. Deprecated, please use
        the enable_async option.

      enable_async: bool Type. Indicates whether async checkpointing is enabled.
        Default is False, i.e., no async checkpoint.

        Async checkpoint moves the checkpoint file writing off the main thread,
        so that the model can continue to train while the checkpoing file
        writing runs in the background. Async checkpoint reduces TPU device idle
        cycles and speeds up model training process, while memory consumption
        may increase.
    