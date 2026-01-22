from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
class CosineDistance(Loss):

    def __init__(self, *, normalize: bool=True, ignore_zeros: bool=False):
        self.normalize = normalize
        self.ignore_zeros = ignore_zeros

    def __call__(self, guesses: Floats2d, truths: Floats2d) -> Tuple[Floats2d, float]:
        return (self.get_grad(guesses, truths), self.get_loss(guesses, truths))

    def get_similarity(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:
            err = f'Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}.'
            raise ValueError(err)
        xp = get_array_module(guesses)
        yh = guesses + 1e-08
        y = truths + 1e-08
        norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
        norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
        mul_norms = norm_yh * norm_y
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
        return cosine

    def get_grad(self, guesses: Floats2d, truths: Floats2d) -> Floats2d:
        if guesses.shape != truths.shape:
            err = f'Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}.'
            raise ValueError(err)
        xp = get_array_module(guesses)
        if self.ignore_zeros:
            zero_indices = xp.abs(truths).sum(axis=1) == 0
        yh = guesses + 1e-08
        y = truths + 1e-08
        norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
        norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
        mul_norms = norm_yh * norm_y
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
        d_yh = y / mul_norms - cosine * (yh / norm_yh ** 2)
        if self.ignore_zeros:
            d_yh[zero_indices] = 0
        if self.normalize:
            d_yh = d_yh / guesses.shape[0]
        return -d_yh

    def get_loss(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:
            err = f'Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}.'
            raise ValueError(err)
        xp = get_array_module(guesses)
        cosine = self.get_similarity(guesses, truths)
        losses = xp.abs(cosine - 1)
        if self.ignore_zeros:
            zero_indices = xp.abs(truths).sum(axis=1) == 0
            losses[zero_indices] = 0
        if self.normalize:
            losses = losses / guesses.shape[0]
        loss = losses.sum()
        return loss