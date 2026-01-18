from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Deci/DeciLM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained("/home/lloyd/Downloads/local_model_store/DeciLM-7B")
model.save_pretrained("/home/lloyd/Downloads/local_model_store/DeciLM-7B")


# Set device based on your hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_path = "/home/lloyd/Downloads/local_model_store/DeciLM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Generate text
prompt = "Once upon a time, in a far-off land"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs, max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=60, temperature=0.9
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
