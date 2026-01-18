import inspect
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel # type: ignore
from accelerate import disk_offload # type: ignore
from smolagents import (
    USER_PROMPT_PLAN_UPDATE,
    CODE_SYSTEM_PROMPT,
    MANAGED_AGENT_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
    SINGLE_STEP_CODE_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION
)

prompts = {
    "USER_PROMPT_PLAN_UPDATE": USER_PROMPT_PLAN_UPDATE,
    "CODE_SYSTEM_PROMPT": CODE_SYSTEM_PROMPT,
    "MANAGED_AGENT_PROMPT": MANAGED_AGENT_PROMPT,
    "TOOL_CALLING_SYSTEM_PROMPT": TOOL_CALLING_SYSTEM_PROMPT,
    "SINGLE_STEP_CODE_SYSTEM_PROMPT": SINGLE_STEP_CODE_SYSTEM_PROMPT,
    "PLAN_UPDATE_FINAL_PLAN_REDACTION": PLAN_UPDATE_FINAL_PLAN_REDACTION,
}

for name, content in prompts.items():
    filename = f"{name}.txt"
    try:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Saved prompt '{name}' to file '{filename}'.")
    except Exception as e:
        print(f"Error saving prompt '{name}' to file '{filename}': {e}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
code_model = TransformersModel(model_name, "cpu")

try:
    with open("CODE_SYSTEM_PROMPT.txt", "r") as f:
        modified_code_prompt = f.read().strip() + "\nDon't forget, you are a powerful python code generating agent!"
except FileNotFoundError:
    modified_code_prompt = "You are a powerful python code generating agent!"
    print("Warning: CODE_SYSTEM_PROMPT.txt not found, using default prompt.")
except Exception as e:
    modified_code_prompt = "You are a powerful python code generating agent!"
    print(f"Error reading CODE_SYSTEM_PROMPT.txt: {e}, using default prompt.")

agent = CodeAgent(tools=[], model=code_model, system_prompt=modified_code_prompt, additional_authorized_imports=[], planning_interval=1, add_base_tools=True)

print("\nAgent's deliberation on prompts:")
for name, content in prompts.items():
    print(f"{name} = ```{content}```")
