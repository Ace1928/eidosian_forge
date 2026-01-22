import random
import unittest
from cirq_ft.linalg.lcu_util import (
class DiscretizeDistributionTest(unittest.TestCase):

    def assertGetDiscretizedDistribution(self, probabilities, epsilon):
        total_probability = sum(probabilities)
        numers, denom, mu = _discretize_probability_distribution(probabilities, epsilon)
        self.assertEqual(sum(numers), denom)
        self.assertEqual(len(numers), len(probabilities))
        self.assertEqual(len(probabilities) * 2 ** mu, denom)
        for i in range(len(numers)):
            self.assertAlmostEqual(numers[i] / denom, probabilities[i] / total_probability, delta=epsilon)
        return (numers, denom)

    def test_fuzz(self):
        random.seed(8)
        for _ in range(100):
            n = random.randint(1, 50)
            weights = [random.random() for _ in range(n)]
            self.assertGetDiscretizedDistribution(weights, 2 ** (-random.randint(1, 20)))

    def test_known_discretizations(self):
        self.assertEqual(self.assertGetDiscretizedDistribution([1], 0.25), ([4], 4))
        self.assertEqual(self.assertGetDiscretizedDistribution([1], 0.125), ([8], 8))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.1, 0.1, 0.1], 0.25), ([2, 2, 2], 6))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.25), ([2, 2, 2], 6))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.1), ([4, 4, 4], 12))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.05), ([7, 9, 8], 24))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.01), ([58, 70, 64], 192))
        self.assertEqual(self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.00335), ([115, 141, 128], 384))