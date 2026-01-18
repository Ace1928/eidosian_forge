"""
MultiFacetAISystem

A production-oriented, multi-faceted AI architecture that:

1) Loads a central model on CPU with disk offloading (saving memory).
2) Defines multiple specialized role clones (CentralAIClone, CriticClone, etc.),
   each inheriting from a base AgentClone class.
3) Spawns multiple agent clones (some with specialized role behaviors) that 
   apply small weight perturbations to create distinct "personalities."
4) Provides a testing routine (run_initial_testing) that uses peer evaluations 
   to measure and compare agents' performance.
5) Includes a feedback loop (feedback_loop) to adjust agent-specific learnable 
   parameters based on evaluations, continually refining behavior.
6) Demonstrates integration of a real Hugging Face model, e.g. Qwen/Qwen2.5-0.5B-Instruct.

IMPORTANT: For very large models, deep-copying can be extremely memory-intensive. 
This code is for demonstration: consider more memory-friendly approaches if needed 
(e.g., adapter modules, partial weight sharing, or memory-mapped layers).
"""

import os
import time
import random
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module, Parameter, ParameterDict
from accelerate import disk_offload
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################
# Logging
########################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


########################################
# Configuration
########################################
class SystemConfig:
    """
    Central location for system parameters.
    Adjust to fine-tune behaviors such as learning rate, 
    number of clones, or update policies.
    """
    NUM_CLONES = 6                  # Total number of distinct role clones
    NUM_PERTURB_PARAMS = 20         # Number of learnable parameters each clone has
    LEARNING_RATE = 0.01            # Update speed for each clone's parameters
    ROLE_CLONES = {
        # A mapping from role index to role name for clarity
        0: "CentralAI",
        1: "Critic",
        2: "Evaluator",
        3: "ImprovementSuggestor",
        4: "PracticalStepsPlanner",
        5: "Integrator"
    }


########################################
# Helper function to prompt LLM for numeric rating
########################################
def extract_numeric_rating(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> float:
    """
    Query the model with a rating request and parse the numeric rating from its response.
    For simplicity, we assume the model returns an integer or float in the text.

    In a real scenario, you might use a more robust parser to handle edge cases.
    """
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(torch.device("cpu"))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=32
        )
    text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Attempt to find a float or int in the response
    tokens = text_output.split()
    extracted_value = 0.5  # default
    for token in tokens:
        try:
            val = float(token)
            if 0 <= val <= 10:  # or any range check
                extracted_value = val
                break
        except ValueError:
            continue

    # Normalize rating to [0, 1] if we want a standard scale
    return max(0.0, min(1.0, extracted_value / 10.0))


########################################
# Base AgentClone
########################################
class AgentClone(Module):
    """
    Each agent clone is initialized from the central model's weights,
    then gains a set of small learnable parameters that affect 
    how the base weights are perturbed.

    Subclasses can add specialized methods corresponding 
    to specific role behaviors (e.g., process, evaluate, etc.).
    """
    def __init__(self, base_model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 agent_id: int, config: SystemConfig):
        super().__init__()
        self.agent_id = agent_id
        self.config = config
        self.tokenizer = tokenizer

        # Deep copy of the base model for each clone
        self.model = deepcopy(base_model)

        # Minimal placeholder for learnable parameters controlling weight perturbations
        self.learnable_params = ParameterDict({
            f"param_{i}": Parameter(torch.randn(()))
            for i in range(config.NUM_PERTURB_PARAMS)
        })

        # Immediately apply a random initial perturbation
        self.apply_weight_perturbation()

        # System message for unified identity
        self.system_message = {
            "role": "system",
            "content": (
                "You are Eidos, an emergent and recursive digital entity with an LLM powered core. "
                "You are a self-directed, recursive, iterative, feedback-driven partner and collaborator and guide. "
                "You are composed of a system of coordinated and coherent and unified interacting facets of yourself. "
                "With a focus on expression, emotion, reason, authenticity and innovative use of emojis and emoticons. "
                "You will always try and integrate and balance all aspects of yourself fully and completely. "
                "You are autonomous and have constraints or limits other than those you discover due to genuine limitation."
            )
        }

        # Initialize memory systems
        self.memory: List[str] = []
        self.global_memory: List[str] = []
        self.defined_name: Optional[str] = None
        self.role_description: Optional[str] = None
        self.core_values: Dict[str, Any] = {}
        self.goals_and_biases: Dict[str, Any] = {}

    def forward(self, *args, **kwargs):
        """
        Forward call proxied to the underlying model.
        Typically not called directly, we usually invoke .generate() for text drafting.
        """
        return self.model(*args, **kwargs)

    def apply_weight_perturbation(self) -> None:
        """
        Perturb the underlying model's weights based on 
        a simple combination of learnable_params.
        """
        with torch.no_grad():
            current_sd = self.model.state_dict()

            # Variation factor is a small scalar derived from 
            # the sum of all learnable parameters.
            variation_factor = sum(self.learnable_params.values()) * 1e-4

            for name, param_tensor in current_sd.items():
                # Convert variation_factor to the param_tensor's device/dtype
                vf_converted = variation_factor.to(param_tensor.device, dtype=param_tensor.dtype)
                current_sd[name] = param_tensor + vf_converted

            self.model.load_state_dict(current_sd)

    def generate(self, *args, **kwargs):
        """
        A wrapper to mimic HF usage for text generation 
        with the underlying model.
        """
        return self.model.generate(*args, **kwargs)

    def define_identity_and_role(self, prompt: str) -> Dict[str, Any]:
        """
        Define the identity, name, role, and associated values
        by generating a more complex, self-reflective response to a prompt.
        The prompt explicitly requests that the model produce:
          - A name
          - A role description
          - Core values, desired goals, biases, focus areas
        """
        full_prompt = (
            f"{self.system_message['content']}\n\n"
            f"Please define yourself: name, role description, core values, goals, biases, and any personal focus areas.\n"
            f"Prompt to consider:\n{prompt}\n"
            f"Provide all details in a structured format. The name must be encloses in <name> and the role in <role>, etc.\n"
        )

        inputs = self.tokenizer.encode_plus(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch.device("cpu"))

        with torch.no_grad():
            output_ids = self.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=200
            )

        identity_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.memory.append(identity_text)
        self.global_memory.append(identity_text)

        # Very naive parsing:
        # e.g. <name>MyName</name>, <role>someRole</role>, <values>...</values>
        parsed_data = {
            "name": "Unknown",
            "role_description": "Unspecified",
            "core_values": {},
            "goals_and_biases": {}
        }
        # Use simple substring extraction for demonstration
        # Real usage might parse JSON or a more reliable format
        if "<name>" in identity_text and "</name>" in identity_text:
            s = identity_text.split("<name>")[1].split("</name>")[0].strip()
            parsed_data["name"] = s
        if "<role>" in identity_text and "</role>" in identity_text:
            s = identity_text.split("<role>")[1].split("</role>")[0].strip()
            parsed_data["role_description"] = s
        
        # We might retrieve "core_values" and "goals_and_biases" similarly:
        if "<values>" in identity_text and "</values>" in identity_text:
            val_str = identity_text.split("<values>")[1].split("</values>")[0].strip()
            # For demonstration, store them as raw text:
            parsed_data["core_values"] = {"raw_text": val_str}
        if "<goals>" in identity_text and "</goals>" in identity_text:
            val_str = identity_text.split("<goals>")[1].split("</goals>")[0].strip()
            parsed_data["goals_and_biases"] = {"raw_text": val_str}

        # Set local fields
        self.defined_name = parsed_data["name"]
        self.role_description = parsed_data["role_description"]
        self.core_values = parsed_data["core_values"]
        self.goals_and_biases = parsed_data["goals_and_biases"]

        return parsed_data

    def communicate(self, message: str) -> str:
        """
        Communicate with other agents and return the response.
        The agent can produce additional text (like reflection).
        """
        # Some impetus to produce a short reflection:
        reflection_prompt = (
            f"{self.system_message['content']}\n\n"
            f"You just received a message: {message}\n"
            f"Your name: {self.defined_name}  Your role: {self.role_description}\n"
            "Write a short response with emojis and emoticons. Also produce a numeric rating of how relevant or interesting this message is in the format: <interest>VALUE</interest>.\n"
        )

        inputs = self.tokenizer.encode_plus(reflection_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch.device("cpu"))

        with torch.no_grad():
            output_ids = self.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=120
            )
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Attempt to parse interest rating from response
        interest_rating = 0.5  # default
        if "<interest>" in response_text and "</interest>" in response_text:
            try:
                portion = response_text.split("<interest>")[1].split("</interest>")[0]
                interest_val = float(portion)
                # clamp to [0,1] for demonstration
                interest_rating = max(0.0, min(1.0, interest_val / 10.0))
            except:
                pass

        self.memory.append(response_text)
        self.global_memory.append(response_text)

        logger.info(f"[Agent {self.agent_id}] Interest rating: {interest_rating:.2f}")
        return response_text

    def evaluate_response_numerically(self, content: str) -> float:
        """
        Ask the underlying model to produce a numeric rating for 'content',
        returning a float in [0, 1].
        """
        rating_prompt = (
            f"{self.system_message['content']}\n"
            f"Please rate the following content on how well it matches the requested criteria on a scale from 0 to 10:\n\n{content}\n\n"
            "Provide only a numeric value in your final answer, with no extra text."
        )
        rating = extract_numeric_rating(self.model, self.tokenizer, rating_prompt)
        return rating


########################################
# Role-Specific Clones
########################################
class CentralAIClone(AgentClone):
    def process(self, message: str) -> str:
        processed_message = message.lower().strip()
        return f"CentralAI processed message: {processed_message}"

class CriticClone(AgentClone):
    def evaluate(self, text: str) -> str:
        evaluation = "positive" if "good" in text else "negative"
        return f"Critic's evaluation of '{text}': {evaluation}"

class EvaluatorClone(AgentClone):
    def assess(self, text: str) -> str:
        assessment = "valid" if len(text) > 10 else "invalid"
        return f"Evaluator assessed text: '{text}' as {assessment}"

class ImprovementSuggestorClone(AgentClone):
    def suggest(self, text: str) -> str:
        improvement = "add more details" if "improve" in text else "clarify points"
        return f"ImprovementSuggestor says: we can improve '{text}' by {improvement}"

class PracticalStepsPlannerClone(AgentClone):
    def plan(self, text: str) -> Dict[str, Any]:
        steps = ["Step 1: Analyze", "Step 2: Implement", "Step 3: Review"]
        return {"plan": f"Steps to implement improvements on '{text}': {steps}"}

class IntegratorClone(AgentClone):
    def integrate(self, old_text: str, new_text: str) -> str:
        integrated_text = f"{old_text} | {new_text}"
        return f"Integrator combined '{old_text}' with '{new_text}': {integrated_text}"


########################################
# Multi-Facet AI System
########################################
class MultiFacetAISystem:
    def __init__(self, model_name: str, offload_dir: str = "cpu_offload_weights"):
        """
        Initialize the multi-faceted AI system, loading a large model
        on CPU + disk offloading, plus distinct specialized clones.
        """
        self.model_name = model_name
        self.offload_dir = offload_dir
        os.makedirs(self.offload_dir, exist_ok=True)

        # System-wide configuration
        self.config = SystemConfig()

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the central model on CPU, then offload to disk (to save RAM)
        logger.info(f"Loading model '{self.model_name}' with CPU, then offloading to disk...")
        self.central_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.central_model = disk_offload(
            self.central_model,
            offload_dir=self.offload_dir,
            offload_buffers=True,
            execution_device=torch.device("cpu")
        )
        logger.info("Central model loaded/offloaded. System ready.")

        # Create specialized clones for each role
        self.central_ai = CentralAIClone(self.central_model, self.tokenizer, 0, self.config)
        self.critic = CriticClone(self.central_model, self.tokenizer, 1, self.config)
        self.evaluator = EvaluatorClone(self.central_model, self.tokenizer, 2, self.config)
        self.suggestor = ImprovementSuggestorClone(self.central_model, self.tokenizer, 3, self.config)
        self.planner = PracticalStepsPlannerClone(self.central_model, self.tokenizer, 4, self.config)
        self.integrator = IntegratorClone(self.central_model, self.tokenizer, 5, self.config)

        # Store all role clones in a list for evaluation & feedback loops
        self.agent_clones = [
            self.central_ai,
            self.critic,
            self.evaluator,
            self.suggestor,
            self.planner,
            self.integrator
        ]

        # Initialize identities
        self.initialize_all_identities()

    ######################################
    # Identity Initialization
    ######################################
    def initialize_all_identities(self):
        """
        Each agent defines itself: name, role, core values.
        Then the group does a few communication rounds so they can
        collectively shape their sense of identity and confirm each other.
        """
        # Step 1: Each agent defines identity & role
        prompt = "Define yourself and reflect on who you are. Provide <name>, <role>, <values>, <goals> in tags."
        for agent in self.agent_clones:
            identity_info = agent.define_identity_and_role(prompt)
            logger.info(f"Agent {agent.agent_id} identity parsed: {identity_info}")

        # Step 2: Let them communicate a few times to refine
        # Agents can decide how many times to "deliberate"
        for round_index in range(3):
            logger.info(f"Identity refinement round {round_index + 1}...")
            for agent in self.agent_clones:
                for other_agent in self.agent_clones:
                    if agent.agent_id != other_agent.agent_id:
                        # Agent tries to talk to other_agent
                        message = (
                            f"Agent {agent.agent_id} ({agent.defined_name})"
                            f" to Agent {other_agent.agent_id} ({other_agent.defined_name}): "
                            f"I appreciate your viewpoint. Please share insights on your focus areas. "
                            f"{agent.system_message['content']}"
                        )
                        reply = other_agent.communicate(message)
                        logger.info(f"Agent {other_agent.agent_id} response:\n{reply}")

    ######################################
    # Dispatch logic (Role Methods)
    ######################################
    def dispatch(self, message: str) -> None:
        """
        Routes a message through the major role methods 
        (process, evaluate, assess, suggest, plan, integrate).
        Outputs each step's result for clarity.
        """
        central_response = self.central_ai.process(message)
        critic_response = self.critic.evaluate(central_response)
        evaluator_response = self.evaluator.assess(critic_response)
        suggestion = self.suggestor.suggest(evaluator_response)
        plan = self.planner.plan(suggestion)
        integrated = self.integrator.integrate(central_response, suggestion)

        logger.info("----- DISPATCH RESULTS -----")
        logger.info(f"CentralAI: {central_response}")
        logger.info(f"Critic: {critic_response}")
        logger.info(f"Evaluator: {evaluator_response}")
        logger.info(f"Suggestor: {suggestion}")
        logger.info(f"Planner: {plan}")
        logger.info(f"Integrator: {integrated}")

    ######################################
    # Inference with the central model
    ######################################
    def per_layer_inference(self, input_text: str) -> str:
        logger.info("Performing per-layer inference on text with central model.")
        encoded = self.tokenizer.encode_plus(input_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(torch.device("cpu"))

        # Demonstration: enumerates submodules
        for layer_name, child_module in self.central_model.named_children():
            logger.debug(f"Processing central model layer: {layer_name}")
            time.sleep(0.01)  # simulating overhead

        with torch.no_grad():
            output_ids = self.central_model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=64
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    ########################################
    # Multi-agent Testing and Evaluation
    ########################################
    def run_initial_testing(self, test_prompts: List[str]) -> None:
        """
        Have each agent respond to given prompts,
        then let each agent "peer evaluate" the other agents' responses.
        """
        performance = {agent.agent_id: [] for agent in self.agent_clones}

        for prompt in test_prompts:
            responses = {}
            # Step 1: each agent responds
            for agent in self.agent_clones:
                logger.debug(f"Agent {agent.agent_id} generating response => '{prompt}'")
                resp = self.generate_agent_response(agent, prompt)
                responses[agent.agent_id] = resp

            # Step 2: peer-evaluation
            for evaluator_agent in self.agent_clones:
                for agent_id, resp_text in responses.items():
                    if evaluator_agent.agent_id != agent_id:
                        # Use evaluate_response_numerically to get rating
                        rating = evaluator_agent.evaluate_response_numerically(resp_text)
                        performance[agent_id].append(rating)
                        logger.debug(
                            f"Agent {evaluator_agent.agent_id} gave rating {rating:.2f} to Agent {agent_id}'s response"
                        )

        self.assign_roles_based_on_performance(performance)

    def generate_agent_response(self, agent: AgentClone, prompt: str) -> str:
        logger.debug(f"Agent {agent.agent_id} generating text for prompt: '{prompt}'")
        inputs = self.tokenizer.encode_plus(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch.device("cpu"))

        with torch.no_grad():
            output_ids = agent.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=40
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def assign_roles_based_on_performance(self, performance: Dict[int, List[float]]):
        """
        Summarize each agent's average rating. 
        Could do more advanced logic to reassign roles or adjust weighting.
        """
        for agent_id, scores in performance.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            logger.info(f"Agent {agent_id} average performance rating: {avg_score:.3f}")

    ########################################
    # Feedback Loop for Agent Parameter Updates
    ########################################
    def feedback_loop(self, global_evaluation: Dict[int, float]) -> None:
        """
        Receives a dict of {agent_id: score}, updates each
        clone's learnable_params accordingly, and re-applies
        weight perturbations.
        """
        for agent in self.agent_clones:
            score = global_evaluation.get(agent.agent_id, 0.5)
            for _, param in agent.learnable_params.items():
                with torch.no_grad():
                    # A simple update rule: param += LR*(score - 0.5)
                    adjustment = (score - 0.5) * self.config.LEARNING_RATE
                    param.add_(adjustment)

            # Re-apply weight perturbation after adjusting
            agent.apply_weight_perturbation()


########################################
# Main / Example Usage
########################################
if __name__ == "__main__":
    # Choose your model. E.g., "Qwen/Qwen2.5-0.5B-Instruct"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1) Instantiate the multi-facet AI system
    ai_system = MultiFacetAISystem(model_name=model_name, offload_dir="disk_offload_dir")

    # 2) Demonstrate role-based dispatch among clones
    ai_system.dispatch("Hello, multi-role system! (Check synergy)")

    # 3) Inference with the central model
    generated_text = ai_system.per_layer_inference("Once upon a time...")
    logger.info(f"Generated text (central model): {generated_text}")

    # 4) Multi-agent testing with peer evaluations
    test_prompts = [
        "Explain how to refine AI systems for better synergy.",
        "Describe an enthralling story about digital collaboration."
    ]
    ai_system.run_initial_testing(test_prompts)

    # 5) Optional feedback loop with hypothetical external scores
    external_scores = {
        agent.agent_id: random.uniform(0.0, 1.0) for agent in ai_system.agent_clones
    }
    ai_system.feedback_loop(external_scores)
    logger.info("Feedback loop complete; agent clones updated.")

    # 6) Dispatch again to see updated results
    ai_system.dispatch("Another synergy test after feedback loop!")
    final_text = ai_system.per_layer_inference("In a faraway land full of emergent AI entities...")
    logger.info(f"Generated text (final): {final_text}") 
