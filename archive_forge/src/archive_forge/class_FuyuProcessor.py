import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
class FuyuProcessor(ProcessorMixin):
    """
    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'FuyuImageProcessor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384
        self.pad_token_id = 0
        self.dummy_image_index = -1

    def _left_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool):
        max_length_input_ids = max((entry['input_ids'].shape[1] for entry in model_inputs))
        max_length_image_patch_indices = max((entry['image_patches_indices'].shape[1] for entry in model_inputs))
        batched_inputs = {'input_ids': [], 'image_patches': [], 'image_patches_indices': [], 'attention_mask': []}
        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == 'input_ids':
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    padded_input_ids = torch.cat([torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long), tensor], dim=1)
                    batched_inputs[key].append(padded_input_ids)
                    attention_mask = torch.cat([torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long), torch.ones_like(tensor)], dim=1)
                    batched_inputs['attention_mask'].append(attention_mask)
                elif key == 'image_patches':
                    batched_inputs[key].append(tensor)
                else:
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    padded_indices = torch.cat([torch.full((tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long), tensor], dim=1)
                    batched_inputs[key].append(padded_indices)
        batched_keys = ['input_ids', 'image_patches_indices']
        if return_attention_mask:
            batched_keys.append('attention_mask')
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)
        return batched_inputs

    def get_sample_encoding(self, prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, image_placeholder_id, image_newline_id, tensor_batch_images):
        image_present = torch.ones(1, 1, 1)
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(image_input=tensor_batch_images, image_present=image_present, image_unpadded_h=image_unpadded_heights, image_unpadded_w=image_unpadded_widths, image_placeholder_id=image_placeholder_id, image_newline_id=image_newline_id, variable_sized=True)
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(tokenizer=self.tokenizer, prompts=prompts, scale_factors=scale_factors, max_tokens_to_generate=self.max_tokens_to_generate, max_position_embeddings=self.max_position_embeddings, add_BOS=True, add_beginning_of_answer_token=True)
        image_padded_unpacked_tokens = construct_full_unpacked_stream(num_real_text_tokens=prompts_length, input_stream=prompt_tokens, image_tokens=model_image_input['image_input_ids'], batch_size=1, num_sub_sequences=self.subsequence_length)
        unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(num_real_text_tokens=prompts_length, input_stream=torch.full_like(prompt_tokens, -1), image_tokens=model_image_input['image_patch_indices_per_batch'], batch_size=1, num_sub_sequences=self.subsequence_length)
        max_prompt_length = max((x.shape[-1] for x in image_padded_unpacked_tokens))
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[0].shape[0]))
        image_patch_input_indices = full_unpacked_stream_to_tensor(all_bi_tokens_to_place=[tokens_to_place], full_unpacked_stream=unpacked_image_patch_indices_per_batch, fill_value=-1, batch_size=1, new_seq_len=max_seq_len_batch, offset=0)
        image_patches_tensor = torch.stack([img[0] for img in model_image_input['image_patches']])
        batch_encoding = {'input_ids': image_padded_unpacked_tokens[0].unsqueeze(0), 'image_patches': image_patches_tensor, 'image_patches_indices': image_patch_input_indices}
        return batch_encoding

    def __call__(self, text=None, images=None, add_special_tokens: bool=True, return_attention_mask: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_token_type_ids: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> 'FuyuBatchFeature':
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ['torch'])
        if not return_attention_mask:
            raise ValueError('`return_attention_mask=False` is not supported for this model.')
        if text is None and images is None:
            raise ValueError('You have to specify either text or images. Both cannot be None.')
        if text is not None and images is None:
            logger.warning('You are processing a text with no associated image. Make sure it is intended.')
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            return text_encoding
        if text is None and images is not None:
            logger.warning('You are processing an image with no associated text. Make sure it is intended.')
            prompts = [['']]
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]
        image_encoding = self.image_processor.preprocess(images, return_tensors='pt')
        batch_images = image_encoding['images']
        image_unpadded_heights = image_encoding['image_unpadded_heights']
        image_unpadded_widths = image_encoding['image_unpadded_widths']
        scale_factors = image_encoding['image_scale_factors']
        self.subsequence_length = 1
        self.batch_size = len(batch_images)
        image_placeholder_id = self.tokenizer('|SPEAKER|', add_special_tokens=False)['input_ids'][1]
        image_newline_id = self.tokenizer('|NEWLINE|', add_special_tokens=False)['input_ids'][1]
        tensor_batch_images = torch.stack([img[0] for img in batch_images]).unsqueeze(1)
        all_encodings = []
        for prompt, scale_factor, image_unpadded_height, image_unpadded_width, tensor_batch_image in zip(prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, tensor_batch_images):
            sample_encoding = self.get_sample_encoding(prompts=[prompt], scale_factors=[scale_factor], image_unpadded_heights=torch.tensor([image_unpadded_height]), image_unpadded_widths=torch.tensor([image_unpadded_width]), image_placeholder_id=image_placeholder_id, image_newline_id=image_newline_id, tensor_batch_images=tensor_batch_image.unsqueeze(0))
            all_encodings.append(sample_encoding)
        batch_encoding = self._left_pad_inputs_with_attention_mask(model_inputs=all_encodings, return_attention_mask=return_attention_mask)
        return FuyuBatchFeature(data=batch_encoding)

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """

        def scale_factor_to_fit(original_size, target_size=None):
            height, width = original_size
            if target_size is None:
                max_height = self.image_processor.size['height']
                max_width = self.image_processor.size['width']
            else:
                max_height, max_width = target_size
            if width <= max_width and height <= max_height:
                return 1.0
            return min(max_height / height, max_width / width)

        def find_delimiters_pair(tokens, start_token, end_token):
            start_id = self.tokenizer.convert_tokens_to_ids(start_token)
            end_id = self.tokenizer.convert_tokens_to_ids(end_token)
            starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
            ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]
            if torch.any(starting_positions) and torch.any(ending_positions):
                return (starting_positions[0], ending_positions[0])
            return (None, None)

        def tokens_to_boxes(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (None, None):
                start, end = pair
                if end != start + 5:
                    continue
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
                scale = scale_factor_to_fit(original_size)
                top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]
                replacement = f' {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}'
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)
                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
            return tokens

        def tokens_to_points(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (None, None):
                start, end = pair
                if end != start + 3:
                    continue
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
                scale = scale_factor_to_fit(original_size)
                x, y = [2 * int(float(c) / scale) for c in coords]
                replacement = f' {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}'
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)
                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
            return tokens
        if target_sizes is None:
            target_sizes = ((self.image_processor.size['height'], self.image_processor.size['width']),) * len(outputs)
        elif target_sizes.shape[1] != 2:
            raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
        if len(outputs) != len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as output sequences')
        results = []
        for seq, size in zip(outputs, target_sizes):
            seq = tokens_to_boxes(seq, size)
            seq = tokens_to_points(seq, size)
            results.append(seq)
        return results

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)