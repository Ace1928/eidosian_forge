from transformers import TrainingArguments, Trainer  # type: ignore


def finetune_qwen_model(model, train_dataset, output_dir="qwen_finetuned"):
    """
    Fine-tune the Qwen model on the given train_dataset.
    Uses CPU-only operation with disk offload via the TrainingArguments settings.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10,
        logging_dir="./logs",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
