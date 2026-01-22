import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
class _InformationMeasure:
    """A wrapper class used for the calculation of different information measures.

    This metric can be used to measure the information between the discrete reference distributions of predicted and
    reference sentences. The class also handles input validation for `alpha` and `beta` parameters.

    Args:
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.

    Raises:
        ValueError:
            If information measure is one from alpha, AB or Rényi divergence and parameter `alpha` is `None`.
        ValueError:
            If information measure is one from beta or divergence and parameter `beta` is `None`.
        ValueError:
            If information measure is alpha divergence and parameter `alpha` equals 0 or 1.
        ValueError:
            If information measure is beta divergence and parameter `beta` equals 0 or -1.
        ValueError:
            If information measure is AB divergence and parameter `alpha`, `beta` or `alpha + beta` equal 0.
        ValueError:
            If information measure is Rényi divergence and parameter `alpha` equals 1.

    """

    def __init__(self, information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL, alpha: Optional[float]=None, beta: Optional[float]=None) -> None:
        self.information_measure = _IMEnum.from_str(information_measure)
        _bad_measures = (_IMEnum.ALPHA_DIVERGENCE, _IMEnum.AB_DIVERGENCE, _IMEnum.RENYI_DIVERGENCE)
        if self.information_measure in _bad_measures and (not isinstance(alpha, float)):
            raise ValueError(f'Parameter `alpha` is expected to be defined for {information_measure}.')
        if self.information_measure in [_IMEnum.BETA_DIVERGENCE, _IMEnum.AB_DIVERGENCE] and (not isinstance(beta, float)):
            raise ValueError(f'Parameter `beta` is expected to be defined for {information_measure}.')
        if self.information_measure == _IMEnum.ALPHA_DIVERGENCE and (not isinstance(alpha, float) or alpha in [0, 1]):
            raise ValueError(f'Parameter `alpha` is expected to be float differened from 0 and 1 for {information_measure}.')
        if self.information_measure == _IMEnum.BETA_DIVERGENCE and (not isinstance(beta, float) or beta in [0, -1]):
            raise ValueError(f'Parameter `beta` is expected to be float differened from 0 and -1 for {information_measure}.')
        if self.information_measure == _IMEnum.AB_DIVERGENCE and (alpha is None or beta is None or (any((not isinstance(p, float) for p in [alpha, beta])) or 0 in [alpha, beta, alpha + beta])):
            raise ValueError(f'Parameters `alpha`, `beta` and their sum are expected to be differened from 0 for {information_measure}.')
        if self.information_measure == _IMEnum.RENYI_DIVERGENCE and (not isinstance(alpha, float) or alpha == 1):
            raise ValueError(f'Parameter `alpha` is expected to be float differened from 1 for {information_measure}.')
        self.alpha = alpha or 0
        self.beta = beta or 0

    def __call__(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        information_measure_function = getattr(self, f'_calculate_{self.information_measure.value}')
        return torch.nan_to_num(information_measure_function(preds_distribution, target_distribution))

    @staticmethod
    def _calculate_kl_divergence(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Kullback-Leibler divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Kullback-Leibler divergence between discrete distributions of predicted and reference sentences.

        """
        return torch.sum(target_distribution * torch.log(preds_distribution / target_distribution), dim=-1)

    def _calculate_alpha_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate alpha divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Alpha divergence between discrete distributions of predicted and reference sentences.

        """
        _alpha_denom = self.alpha * (self.alpha - 1)
        return (1 - torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)) / _alpha_denom

    def _calculate_ab_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate AB divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            AB divergence between discrete distributions of predicted and reference sentences.

        """
        a = torch.log(torch.sum(target_distribution ** (self.beta + self.alpha), dim=-1))
        a /= self.beta * (self.beta + self.alpha)
        b = torch.log(torch.sum(preds_distribution ** (self.beta + self.alpha), dim=-1))
        b /= self.alpha * (self.beta + self.alpha)
        c = torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** self.beta, dim=-1))
        c /= self.alpha * self.beta
        return a + b - c

    def _calculate_beta_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate beta divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Beta divergence between discrete distributions of predicted and reference sentences.

        """
        self.alpha = 1.0
        return self._calculate_ab_divergence(preds_distribution, target_distribution)

    def _calculate_renyi_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Rényi divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Rényi divergence between discrete distributions of predicted and reference sentences.

        """
        return torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)

    @staticmethod
    def _calculate_l1_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L1 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L1 distance between discrete distributions of predicted and reference sentences.

        """
        return torch.norm(target_distribution - preds_distribution, p=1, dim=-1)

    @staticmethod
    def _calculate_l2_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L2 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L2 distance between discrete distributions of predicted and reference sentences.

        """
        return torch.norm(target_distribution - preds_distribution, p=2, dim=-1)

    @staticmethod
    def _calculate_l_infinity_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L-infinity distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L-infinity distance between discrete distributions of predicted and reference sentences.

        """
        return torch.norm(target_distribution - preds_distribution, p=float('inf'), dim=-1)

    @staticmethod
    def _calculate_fisher_rao_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Fisher-Rao distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Fisher-Rao distance between discrete distributions of predicted and reference sentences.

        """
        return 2 * torch.acos(torch.clamp(torch.sqrt(preds_distribution * target_distribution).sum(-1), 0, 1))