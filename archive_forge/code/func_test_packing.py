import pytest
from unittest.mock import patch
@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_packing(step_lr, optimizer, get_model, tokenizer, train, setup_tokenizer, llama_version):
    from llama_recipes.finetuning import main
    setup_tokenizer(tokenizer)
    kwargs = {'model_name': llama_version, 'batch_size_training': 8, 'val_batch_size': 1, 'use_peft': False, 'dataset': 'samsum_dataset', 'batching_strategy': 'packing'}
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    assert len(train_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]['train']
    assert len(eval_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]['eval']
    batch = next(iter(train_dataloader))
    assert 'labels' in batch.keys()
    assert 'input_ids' in batch.keys()
    assert 'attention_mask' in batch.keys()
    assert batch['labels'][0].size(0) == 4096
    assert batch['input_ids'][0].size(0) == 4096
    assert batch['attention_mask'][0].size(0) == 4096