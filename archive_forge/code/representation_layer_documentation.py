from typing import Optional
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
Produces a discrete, differentiable z-sample from some 1D input tensor.

        Pushes the input_ tensor through our dense layer, which outputs
        32(B=num categoricals)*32(c=num classes) logits. Logits are used to:

        1) sample stochastically
        2) compute probs (via softmax)
        3) make sure the sampling step is differentiable (see [2] Algorithm 1):
            sample=one_hot(draw(logits))
            probs=softmax(logits)
            sample=sample + probs - stop_grad(probs)
            -> Now sample has the gradients of the probs.

        Args:
            inputs: The input to our z-generating layer. This might be a) the combined
                (concatenated) outputs of the (image?) encoder + the last hidden
                deterministic state, or b) the output of the dynamics predictor MLP
                network.

        Returns:
            Tuple consisting of a differentiable z-sample and the probabilities for the
            categorical distribution (in the shape of [B, num_categoricals,
            num_classes]) that created this sample.
        