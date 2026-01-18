import pytest
from unittest.mock import patch
from transformers import LlamaTokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_unknown_dataset_error(step_lr, optimizer, tokenizer, get_model, train, mocker):
    from llama_recipes.finetuning import main
    tokenizer.return_value = mocker.MagicMock(side_effect=lambda x: {'input_ids': [len(x) * [0]], 'attention_mask': [len(x) * [0]]})
    kwargs = {'dataset': 'custom_dataset', 'custom_dataset.file': 'recipes/finetuning/datasets/custom_dataset.py:get_unknown_dataset', 'batch_size_training': 1, 'use_peft': False}
    with pytest.raises(AttributeError):
        main(**kwargs)