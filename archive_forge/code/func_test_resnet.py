import torch
from torchvision.models import resnet34
from transformers import (
from accelerate import PartialState
from accelerate.inference import prepare_pippy
from accelerate.utils import DistributedType, send_to_device, set_seed
def test_resnet(batch_size: int=2):
    set_seed(42)
    state = PartialState()
    model = resnet34()
    input_tensor = torch.rand(batch_size, 3, 224, 224)
    model = prepare_pippy(model, example_args=(input_tensor,))
    inputs = send_to_device(input_tensor, 'cuda:0')
    with torch.no_grad():
        output = model(inputs)
    if not state.is_last_process:
        assert output is None, 'Output was not generated on just the last process!'
    else:
        assert output is not None, 'Output was not generated in the last process!'