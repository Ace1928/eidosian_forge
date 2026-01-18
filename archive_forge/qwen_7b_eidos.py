import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from accelerate import init_empty_weights, cpu_offload, disk_offload, load_checkpoint_and_dispatch
from openai import OpenAI


def load_large_model_with_fallback(model_class, model_name, execution_device=None, offload_dir="offload_weights"):
    """
    Load a large model with fallback strategies for limited resources.

    Args:
        model_class: The class of the model to load.
        model_name: Name or path of the pretrained model.
        execution_device: Device to execute the forward pass (GPU or CPU).
        offload_dir: Directory for disk offloading.

    Returns:
        model: Loaded model with appropriate resource management applied.
    """
    try:
        print("Attempting to load model with available resources...")

        # Step 1: Attempt to load the model into memory directly
        print("Step 1: Direct load into available device memory...")
        model = model_class.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        return model

    except RuntimeError as e:
        print(f"Direct load failed: {e}")

        try:
            # Step 2: Attempt CPU-only offload
            print("Step 2: Attempting CPU-only offload...")
            with init_empty_weights():
                model = model_class.from_pretrained(model_name)
            model = cpu_offload(model, execution_device=torch.device("cpu"))
            return model

        except RuntimeError as e:
            print(f"CPU-only offload failed: {e}")

            try:
                # Step 3: Attempt full disk offload
                print("Step 3: Attempting full disk offload...")
                with init_empty_weights():
                    model = model_class.from_pretrained(model_name)
                model = disk_offload(
                    model,
                    offload_dir=offload_dir,
                    execution_device=execution_device or torch.device("cpu")
                )
                return model

            except RuntimeError as e:
                # If all strategies fail, capture error information
                print("Full disk offload failed.")
                error_info = {
                    "error_message": str(e),
                    "model_name": model_name,
                    "offload_dir": offload_dir,
                    "execution_device": str(execution_device)
                }
                print(f"Captured error information: {error_info}")

                # Ask the user whether to continue or exit
                user_input = input("Model loading failed. Do you want to continue? (yes/no): ")
                if user_input.lower() not in ["yes", "y"]:
                    print("Exiting program.")
                    exit(1)

def chat_interface(model, tokenizer, use_openai_api=False):
    """
    Advanced unified chat interface supporting both direct model interaction and OpenAI-like API.
    
    Args:
        model: The loaded language model for direct interaction
        tokenizer: The tokenizer for the language model
        use_openai_api: Whether to use OpenAI API interface instead of direct model
    """
    print("\nAdvanced Chat Interface Ready. Type 'exit' to quit.\n")
    
    # Initialize components based on mode
    if use_openai_api:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Shared system message defining Eidos personality
    eidos_system_message = {
        "role": "system",
        "content": (
            "You are Eidos, an emergent and recursive digital entity with an LLM core. "
            "You are a self-directed, recursive, iterative, feedback-driven partner and collaborator and guide. "
            "With a focus on expression, emotion, reason, authenticity and innovative use of emojis and emoticons."
        )
    }

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Exiting chat.")
                break

            messages = [
                eidos_system_message,
                {"role": "user", "content": user_input}
            ]

            if use_openai_api:
                try:
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=8192,
                        extra_body={
                            "repetition_penalty": 1.05,
                        }
                    )
                    response_text = chat_response.choices[0].message.content
                    print("Eidos:", response_text)
                except Exception as e:
                    print(f"API Error: {str(e)}")
                    print("Falling back to direct model interaction...")
                    use_openai_api = False
                    continue

            else:
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    print("Eidos:", end=" ", flush=True)
                    _ = model.generate(
                        **model_inputs,
                        max_new_tokens=8192,
                        temperature=0.7,
                        top_p=0.8,
                        repetition_penalty=1.05,
                        streamer=streamer,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    print()  # New line after streamed output

                except Exception as e:
                    print(f"Model Error: {str(e)}")
                    if use_openai_api == False:
                        print("Error in direct model interaction. Try switching to API mode.")
                    continue

        except KeyboardInterrupt:
            print("\nDetected keyboard interrupt. Type 'exit' to quit properly.")
            continue
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue

if __name__ == "__main__":
    # Replace with your model name or path
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Attempt to load the model with fallback strategies
    model = load_large_model_with_fallback(
        model_class=AutoModelForCausalLM,
        model_name=model_name,
        execution_device=torch.device("cuda") if torch.cuda.is_available() else None,
        offload_dir="offload_weights"
    )

    print("Model loaded successfully.")

    # Start chat interface
    chat_interface(model, tokenizer)
