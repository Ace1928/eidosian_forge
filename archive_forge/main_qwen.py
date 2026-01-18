from pathlib import Path
from intelligent_tokenizer import DynamicTokenizer
from custom_tokenizer import DynamicPreTrainedTokenizer
from transformer_adapter import update_qwen_model_with_dynamic_tokenizer
from finetune_qwen import finetune_qwen_model
from colorama import Fore, Style, Back
import logging

logger = logging.getLogger(__name__)


def main():
    # Use a persistence file that reflects the persistence_prefix.
    persist_file = Path("intelligent_tokenizer.vocab")

    if persist_file.exists():
        logger.info(
            "Persistent vocabulary found; using it and skipping dynamic rebuild."
        )
        dynamic_rebuild = False
    else:
        logger.info(
            "No persistent vocabulary found; will learn vocabulary from sample text."
        )
        dynamic_rebuild = True

    # Create the dynamic tokenizer using the persistent vocabulary if available,
    # and pass the unique special tokens to its configuration.
    tokenizer = DynamicTokenizer(
        normalization_form="NFC",
        unicode_strategy="extensive",
        category_profile="all",
        sort_mode="unicode",
        dynamic_rebuild=dynamic_rebuild,  # If False, the stored vocabulary will be used.
        persistence_prefix="intelligent_tokenizer",
    )

    if dynamic_rebuild:
        # Let the tokenizer learn from some sample text in background only when needed.
        sample_text = "A sample evolving text with new tokens: 🚀✨"
        tokenizer.learn_in_background(sample_text)

    # Wrap the dynamic tokenizer in the Hugging Face–compatible tokenizer.
    hf_tokenizer = DynamicPreTrainedTokenizer(dynamic_tokenizer=tokenizer)

    # Ensure pad token is in the vocabulary. This is needed for proper batch padding.
    if "[PAD]" not in tokenizer.vocab:
        tokenizer.add_token("[PAD]")
    hf_tokenizer.pad_token = "[PAD]"
    hf_tokenizer.pad_token_id = tokenizer.vocab["[PAD]"]

    # Adapt the Qwen model to the new tokenizer by updating its vocab size and embeddings.
    model = update_qwen_model_with_dynamic_tokenizer(
        tokenizer, model_name="Qwen/Qwen2.5-0.5b"
    )

    # For demonstration, we create a small dummy dataset.
    # Here we use the "wikitext-2-raw-v1" subset from the Hugging Face datasets library.
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize_function(example):
        # In batched mode, example["text"] is a list, so we tokenize each text individually.
        return {"input_ids": [hf_tokenizer.encode(text) for text in example["text"]]}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    logger.info("Starting fine-tuning on WikiText dataset...")
    # Pass hf_tokenizer so we can build a padding data-collator.
    finetune_qwen_model(model, tokenized_dataset, output_dir="qwen_finetuned")
    logger.info("Fine-tuning complete and model saved.")

    # Launch an interactive chat mode.
    print("\nEntering interactive chat mode (type 'exit' to quit).")
    while True:
        user_prompt = input("User: ")
        if user_prompt.lower() in ["exit", "quit"]:
            break
        inputs = hf_tokenizer(user_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        response = hf_tokenizer.decode(outputs[0].tolist())
        print("Bot:", response)


if __name__ == "__main__":
    main()
