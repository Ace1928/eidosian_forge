import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model

        Test if the transformers kwargs are correctly passed
        Here we check that loading a model in half precision works as expected, i.e. the weights of
        the `pretrained_model` attribute is loaded in half precision and you can run a dummy
        forward pass without any issue.
        