from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootCodeyCheckpoint(_messages.Message):
    """Describes a sample at a checkpoint for post-processing.

  Enums:
    PostInferenceStepValueValuesEnum: Postprocessor run that yielded this
      checkpoint.

  Fields:
    codeyTruncatorMetadata: Metadata that describes what was truncated at this
      checkpoint.
    currentSample: Current state of the sample after truncator.
    postInferenceStep: Postprocessor run that yielded this checkpoint.
  """

    class PostInferenceStepValueValuesEnum(_messages.Enum):
        """Postprocessor run that yielded this checkpoint.

    Values:
      STEP_POST_PROCESSING_STEP_UNSPECIFIED: <no description>
      STEP_ORIGINAL_MODEL_OUTPUT: Original model outputs as-is.
      STEP_MODEL_OUTPUT_DEDUPLICATION: Original model outputs after
        deduplication.
      STEP_STOP_SEQUENCE_TRUNCATION: StopSequencePostProcessor.
      STEP_HEURISTIC_TRUNCATION: Heuristic SuffixTruncator step.
      STEP_WALD_TRUNCATION: Go service post-processor.
      STEP_WHITESPACE_TRUNCATION: Truncate trailing whitespace and filter
        whitespace-only completions.
      STEP_FINAL_DEDUPLICATION: Deduplicate after all truncations.
      STEP_TOXICITY_CHECK: Toxicity returns true.
      STEP_RECITATION_CHECK: Recitation causes BLOCK.
      STEP_RETURNED: Return the response to the API.
      STEP_WALKBACK_CORRECTION: Correcting walkback constraint (samples are
        dropped if they don't match the prefix constraint).
      STEP_SCORE_THRESHOLDING: Thresholding samples based on a minimum score.
      STEP_MODEL_CONFIG_STOP_SEQUENCE_TRUNCATION: StopSequencePostProcessor.
      STEP_CUSTOM_STOP_SEQUENCE_TRUNCATION: StopSequencePostProcessor.
      STEP_EXPECTED_SAMPLE_SIZE: Drop extra number of samples that exceeds
        expected_samples.
      STEP_TREE_TRIM_TRUNCATION: Truncated by highest end token score.
    """
        STEP_POST_PROCESSING_STEP_UNSPECIFIED = 0
        STEP_ORIGINAL_MODEL_OUTPUT = 1
        STEP_MODEL_OUTPUT_DEDUPLICATION = 2
        STEP_STOP_SEQUENCE_TRUNCATION = 3
        STEP_HEURISTIC_TRUNCATION = 4
        STEP_WALD_TRUNCATION = 5
        STEP_WHITESPACE_TRUNCATION = 6
        STEP_FINAL_DEDUPLICATION = 7
        STEP_TOXICITY_CHECK = 8
        STEP_RECITATION_CHECK = 9
        STEP_RETURNED = 10
        STEP_WALKBACK_CORRECTION = 11
        STEP_SCORE_THRESHOLDING = 12
        STEP_MODEL_CONFIG_STOP_SEQUENCE_TRUNCATION = 13
        STEP_CUSTOM_STOP_SEQUENCE_TRUNCATION = 14
        STEP_EXPECTED_SAMPLE_SIZE = 15
        STEP_TREE_TRIM_TRUNCATION = 16
    codeyTruncatorMetadata = _messages.MessageField('LearningGenaiRootCodeyTruncatorMetadata', 1)
    currentSample = _messages.StringField(2)
    postInferenceStep = _messages.EnumField('PostInferenceStepValueValuesEnum', 3)