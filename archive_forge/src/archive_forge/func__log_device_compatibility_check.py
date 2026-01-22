import itertools
from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
def _log_device_compatibility_check(policy_name, gpu_details_list):
    """Logs a compatibility check if the devices support the policy.

  Currently only logs for the policy mixed_float16.

  Args:
    policy_name: The name of the dtype policy.
    gpu_details_list: A list of dicts, one dict per GPU. Each dict
      is the device details for a GPU, as returned by
      `tf.config.experimental.get_device_details()`.
  """
    if policy_name != 'mixed_float16':
        return
    supported_device_strs = []
    unsupported_device_strs = []
    for details in gpu_details_list:
        name = details.get('device_name', 'Unknown GPU')
        cc = details.get('compute_capability')
        if cc:
            device_str = '%s, compute capability %s.%s' % (name, cc[0], cc[1])
            if cc >= (7, 0):
                supported_device_strs.append(device_str)
            else:
                unsupported_device_strs.append(device_str)
        else:
            unsupported_device_strs.append(name + ', no compute capability (probably not an Nvidia GPU)')
    if unsupported_device_strs:
        warning_str = _COMPAT_CHECK_WARNING_PREFIX + '\n'
        if supported_device_strs:
            warning_str += 'Some of your GPUs may run slowly with dtype policy mixed_float16 because they do not all have compute capability of at least 7.0. Your GPUs:\n'
        elif len(unsupported_device_strs) == 1:
            warning_str += 'Your GPU may run slowly with dtype policy mixed_float16 because it does not have compute capability of at least 7.0. Your GPU:\n'
        else:
            warning_str += 'Your GPUs may run slowly with dtype policy mixed_float16 because they do not have compute capability of at least 7.0. Your GPUs:\n'
        for device_str in _dedup_strings(supported_device_strs + unsupported_device_strs):
            warning_str += '  ' + device_str + '\n'
        warning_str += 'See https://developer.nvidia.com/cuda-gpus for a list of GPUs and their compute capabilities.\n'
        warning_str += _COMPAT_CHECK_WARNING_SUFFIX
        tf_logging.warning(warning_str)
    elif not supported_device_strs:
        tf_logging.warning('%s\nThe dtype policy mixed_float16 may run slowly because this machine does not have a GPU. Only Nvidia GPUs with compute capability of at least 7.0 run quickly with mixed_float16.\n%s' % (_COMPAT_CHECK_WARNING_PREFIX, _COMPAT_CHECK_WARNING_SUFFIX))
    elif len(supported_device_strs) == 1:
        tf_logging.info('%s\nYour GPU will likely run quickly with dtype policy mixed_float16 as it has compute capability of at least 7.0. Your GPU: %s' % (_COMPAT_CHECK_OK_PREFIX, supported_device_strs[0]))
    else:
        tf_logging.info('%s\nYour GPUs will likely run quickly with dtype policy mixed_float16 as they all have compute capability of at least 7.0' % _COMPAT_CHECK_OK_PREFIX)