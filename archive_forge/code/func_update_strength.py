from lognormal_around import lognormal_around
import random
def update_strength(self, pre_activity, post_activity, post_neuron):
    """
        Update the connection strength based on neuron activity, considering threshold crossings.
        """
    learning_rate = 0.01 * self.global_scaling_factor
    strength_change = pre_activity * post_activity * learning_rate
    if post_neuron.output >= post_neuron.over_threshold:
        strength_change *= -1
    if post_neuron.output >= post_neuron.excitotoxic_threshold:
        self.excitotoxic_inversion_steps = random.randint(20, 40)
    if self.excitotoxic_inversion_steps > 0:
        strength_change *= -1
        self.excitotoxic_inversion_steps -= 1
    self.strength += strength_change
    self.strength = max(0, self.strength)