import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import disk_offload

messages = [{"role": "user", "content": "Who are you?"}]
model_names = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
]

save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

for model_name in model_names:
    model_path = os.path.join(save_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    # Load model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save model and tokenizer
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    # Test the model
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    test_response = pipe(messages)
    print(f"Test response for {model_name}: {test_response}")