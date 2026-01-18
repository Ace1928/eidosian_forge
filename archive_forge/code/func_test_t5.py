import torch
from torchvision.models import resnet34
from transformers import (
from accelerate import PartialState
from accelerate.inference import prepare_pippy
from accelerate.utils import DistributedType, send_to_device, set_seed
def test_t5(batch_size: int=2):
    set_seed(42)
    state = PartialState()
    model, inputs = get_model_and_data_for_text('t5', 'cpu', batch_size)
    example_inputs = {'input_ids': inputs, 'decoder_input_ids': inputs}
    model = prepare_pippy(model, no_split_module_classes=model._no_split_modules, example_kwargs=example_inputs)
    inputs = send_to_device(example_inputs, 'cuda:0')
    with torch.no_grad():
        output = model(*inputs.values())
    if not state.is_last_process:
        assert output is None, 'Output was not generated on just the last process!'
    else:
        assert output is not None, 'Output was not generated in the last process!'