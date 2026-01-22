import torch
from torch import Tensor
from typing_extensions import Literal
Calculatees `Fleiss kappa`_ a statistical measure for inter agreement between raters.

    .. math::
        \kappa = \frac{\bar{p} - \bar{p_e}}{1 - \bar{p_e}}

    where :math:`\bar{p}` is the mean of the agreement probability over all raters and :math:`\bar{p_e}` is the mean
    agreement probability over all raters if they were randomly assigned. If the raters are in complete agreement then
    the score 1 is returned, if there is no agreement among the raters (other than what would be expected by chance)
    then a score smaller than 0 is returned.

    Args:
        ratings: Ratings of shape [n_samples, n_categories] or [n_samples, n_categories, n_raters] depedenent on `mode`.
            If `mode` is `counts`, `ratings` must be integer and contain the number of raters that chose each category.
            If `mode` is `probs`, `ratings` must be floating point and contain the probability/logits that each rater
            chose each category.
        mode: Whether `ratings` will be provided as counts or probabilities.

    Example:
        >>> # Ratings are provided as counts
        >>> import torch
        >>> from torchmetrics.functional.nominal import fleiss_kappa
        >>> _ = torch.manual_seed(42)
        >>> ratings = torch.randint(0, 10, size=(100, 5)).long()  # 100 samples, 5 categories, 10 raters
        >>> fleiss_kappa(ratings)
        tensor(0.0089)

    Example:
        >>> # Ratings are provided as probabilities
        >>> import torch
        >>> from torchmetrics.functional.nominal import fleiss_kappa
        >>> _ = torch.manual_seed(42)
        >>> ratings = torch.randn(100, 5, 10).softmax(dim=1)  # 100 samples, 5 categories, 10 raters
        >>> fleiss_kappa(ratings, mode='probs')
        tensor(-0.0105)

    