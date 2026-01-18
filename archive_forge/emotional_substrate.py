# Emotional Substrate Module
from enum import Enum
from dataclasses import dataclass
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Defines core emotional states and their heuristic influence."""
    CURIOSITY = {"exploration_factor": 1.5, "focus_factor": 0.5}
    FRUSTRATION = {"exploration_factor": 0.5, "focus_factor": 2.0}
    JOY = {"exploration_factor": 1.2, "focus_factor": 1.2}
    DETERMINATION = {"exploration_factor": 1.0, "focus_factor": 1.5}

@dataclass
class EmotionalMetrics:
    """Tracks emotional dynamics and influences."""
    state: EmotionalState
    intensity: float  # Intensity of the current emotional state
    adaptability: float  # How emotions shift with feedback
    resonance: float  # Influence of external factors

    def adapt(self, outcome: float):
        """Adjust emotional intensity and resonance based on feedback."""
        self.intensity += outcome * self.adaptability
        self.resonance = max(0.0, self.resonance + random.uniform(-0.1, 0.1))

class EmotionalSubstrate:
    """Handles emotional dynamics within the Fractal Intelligence framework."""
    def __init__(self, initial_state: EmotionalState, adaptability: float = 0.1):
        self.current_state = EmotionalMetrics(
            state=initial_state,
            intensity=1.0,
            adaptability=adaptability,
            resonance=0.5
        )

    def influence_decision(self, value: float) -> float:
        """Modulates decision-making based on emotional state."""
        factors = self.current_state.state.value
        modified_value = value * factors["exploration_factor"] * self.current_state.intensity
        return modified_value

    def update_emotional_state(self, feedback: float):
        """Adjusts emotional state based on feedback."""
        self.current_state.adapt(feedback)
        self._random_shift()

    def _random_shift(self):
        """Chaos-driven adjustment to simulate emotional variability."""
        possible_states = list(EmotionalState)
        if random.random() < 0.1:  # 10% chance of emotional shift
            self.current_state.state = random.choice(possible_states)
            logger.info(f"Emotional shift: {self.current_state.state.name}")

# Integration Example
if __name__ == "__main__":
    emotional_system = EmotionalSubstrate(EmotionalState.CURIOSITY)
    decision_value = 1.0

    for iteration in range(10):
        decision_value = emotional_system.influence_decision(decision_value)
        feedback = random.uniform(-1, 1)  # Simulated feedback
        emotional_system.update_emotional_state(feedback)
        print(f"Iteration {iteration}: Value={decision_value}, State={emotional_system.current_state.state.name}, Intensity={emotional_system.current_state.intensity}")
