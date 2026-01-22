import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _get_scores_rouge_n(self, all_hypothesis, all_references):
    """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
    metrics = [metric for metric in self.metrics if metric.split('-')[-1].isdigit()]
    if self.apply_avg or self.apply_best:
        scores = {metric: {stat: 0.0 for stat in Rouge.STATS} for metric in metrics}
    else:
        scores = {metric: [{stat: [] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))] for metric in metrics}
    for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
        assert isinstance(hypothesis, str)
        has_multiple_references = False
        if isinstance(references, list):
            has_multiple_references = len(references) > 1
            if not has_multiple_references:
                references = references[0]
        hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
        references = [self._preprocess_summary_as_a_whole(reference) for reference in references] if has_multiple_references else [self._preprocess_summary_as_a_whole(references)]
        for metric in metrics:
            suffix = metric.split('-')[-1]
            n = int(suffix)
            if self.apply_avg:
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0
                for reference in references:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                    total_hypothesis_ngrams_count += hypothesis_count
                    total_reference_ngrams_count += reference_count
                    total_ngrams_overlapping_count += overlapping_ngrams
                score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha)
                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            elif self.apply_best:
                best_current_score = None
                for reference in references:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    if best_current_score is None or score['r'] > best_current_score['r']:
                        best_current_score = score
                for stat in Rouge.STATS:
                    scores[metric][stat] += best_current_score[stat]
            else:
                for reference in references:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    for stat in Rouge.STATS:
                        scores[metric][sample_id][stat].append(score[stat])
    if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
        for metric in metrics:
            for stat in Rouge.STATS:
                scores[metric][stat] /= len(all_hypothesis)
    return scores