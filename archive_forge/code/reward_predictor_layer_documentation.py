from ray.rllib.utils.framework import try_import_tf
Computes the expected reward using N equal sized buckets of possible values.

        Args:
            inputs: The input tensor for the layer, which computes the reward bucket
                weights (logits). [B, dim].

        Returns:
            A tuple consisting of the expected rewards and the logits that parameterize
            the tfp `FiniteDiscrete` distribution object. To get the individual bucket
            probs, do `[FiniteDiscrete object].probs`.
        