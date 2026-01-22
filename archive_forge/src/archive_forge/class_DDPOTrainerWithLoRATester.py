import gc
import unittest
import torch
from trl import is_diffusers_available, is_peft_available
from .testing_utils import require_diffusers
@require_diffusers
class DDPOTrainerWithLoRATester(DDPOTrainerTester):
    """
    Test the DDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(num_epochs=2, train_gradient_accumulation_steps=1, per_prompt_stat_tracking_buffer_size=32, sample_num_batches_per_epoch=2, sample_batch_size=2, mixed_precision=None, save_freq=1000000)
        pretrained_model = 'hf-internal-testing/tiny-stable-diffusion-torch'
        pretrained_revision = 'main'
        pipeline = DefaultDDPOStableDiffusionPipeline(pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=True)
        self.trainer = DDPOTrainer(self.ddpo_config, scorer_function, prompt_function, pipeline)
        return super().setUp()