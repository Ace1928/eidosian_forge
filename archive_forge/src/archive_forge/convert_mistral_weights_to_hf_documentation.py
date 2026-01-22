import argparse
import gc
import json
import os
import shutil
import warnings
import torch
from transformers import (
from transformers import MistralForCausalLM, LlamaTokenizer

Sample usage:

```
python src/transformers/models/mistral/convert_mistral_weights_to_hf.py     --input_dir /path/to/downloaded/mistral/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import MistralForCausalLM, LlamaTokenizer

model = MistralForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
