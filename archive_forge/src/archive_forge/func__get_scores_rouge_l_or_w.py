import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _get_scores_rouge_l_or_w(self, all_hypothesis, all_references, use_w=False):
    """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
    metric = 'rouge-w' if use_w else 'rouge-l'
    if self.apply_avg or self.apply_best:
        scores = {metric: {stat: 0.0 for stat in Rouge.STATS}}
    else:
        scores = {metric: [{stat: [] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))]}
    for sample_id, (hypothesis_sentences, references_sentences) in enumerate(zip(all_hypothesis, all_references)):
        assert isinstance(hypothesis_sentences, str)
        has_multiple_references = False
        if isinstance(references_sentences, list):
            has_multiple_references = len(references_sentences) > 1
            if not has_multiple_references:
                references_sentences = references_sentences[0]
        hypothesis_sentences = self._preprocess_summary_per_sentence(hypothesis_sentences)
        references_sentences = [self._preprocess_summary_per_sentence(reference) for reference in references_sentences] if has_multiple_references else [self._preprocess_summary_per_sentence(references_sentences)]
        if self.apply_avg:
            total_hypothesis_ngrams_count = 0
            total_reference_ngrams_count = 0
            total_ngrams_overlapping_count = 0
            for reference_sentences in references_sentences:
                hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                total_hypothesis_ngrams_count += hypothesis_count
                total_reference_ngrams_count += reference_count
                total_ngrams_overlapping_count += overlapping_ngrams
            score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha, self.weight_factor)
            for stat in Rouge.STATS:
                scores[metric][stat] += score[stat]
        elif self.apply_best:
            best_current_score = None
            best_current_score_wlcs = None
            for reference_sentences in references_sentences:
                hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)
                if use_w:
                    reference_count_for_score = reference_count ** (1.0 / self.weight_factor)
                    overlapping_ngrams_for_score = overlapping_ngrams
                    score_wlcs = (overlapping_ngrams_for_score / reference_count_for_score) ** (1.0 / self.weight_factor)
                    if best_current_score_wlcs is None or score_wlcs > best_current_score_wlcs:
                        best_current_score = score
                        best_current_score_wlcs = score_wlcs
                elif best_current_score is None or score['r'] > best_current_score['r']:
                    best_current_score = score
            for stat in Rouge.STATS:
                scores[metric][stat] += best_current_score[stat]
        else:
            for reference_sentences in references_sentences:
                hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)
                for stat in Rouge.STATS:
                    scores[metric][sample_id][stat].append(score[stat])
    if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
        for stat in Rouge.STATS:
            scores[metric][stat] /= len(all_hypothesis)
    return scores