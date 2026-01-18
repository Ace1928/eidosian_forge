from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
Compute the `Tweedie Deviance Score`_.

    .. math::
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }p=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }p=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }p=2\\
        2 * (\frac{(max(y,0))^{2 - p}}{(1 - p)(2 - p)} - \frac{y(\hat{y})^{1 - p}}{1 - p} + \frac{(
            \hat{y})^{2 - p}}{2 - p}), & \text{otherwise}
        \end{cases}

    where :math:`y` is a tensor of targets values, :math:`\hat{y}` is a tensor of predictions, and
    :math:`p` is the `power`.

    Args:
        preds: Predicted tensor with shape ``(N,...)``
        targets: Ground truth tensor with shape ``(N,...)``
        power:
            - `power < 0` : Extreme stable distribution. (Requires: preds > 0.)
            - `power = 0` : Normal distribution. (Requires: targets and preds can be any real numbers.)
            - `power = 1` : Poisson distribution. (Requires: targets >= 0 and y_pred > 0.)
            - `1 < p < 2` : Compound Poisson distribution. (Requires: targets >= 0 and preds > 0.)
            - `power = 2` : Gamma distribution. (Requires: targets > 0 and preds > 0.)
            - `power = 3` : Inverse Gaussian distribution. (Requires: targets > 0 and preds > 0.)
            - `otherwise` : Positive stable distribution. (Requires: targets > 0 and preds > 0.)

    Example:
        >>> from torchmetrics.functional.regression import tweedie_deviance_score
        >>> targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
        >>> tweedie_deviance_score(preds, targets, power=2)
        tensor(1.2083)

    