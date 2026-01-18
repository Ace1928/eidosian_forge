import torch

tensor_factory_functions defines the list of torch functions that create tensors.
The list is grabbed by searching thru native_functions.yaml by the following
regular expression:

  cat native_functions.yaml | grep 'func:' | grep -v "Tensor.*->" | grep "[-]>.*Tensor"

It's possible that new tensor factory functions are added making this list stale.
Use at your own risk or regenerate the list.
