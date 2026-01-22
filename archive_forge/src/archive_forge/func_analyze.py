import os
@staticmethod
def analyze(model_path=None, model_content=None, gpu_compatibility=False, **kwargs):
    """Analyzes the given tflite_model with dumping model structure.

    This tool provides a way to understand users' TFLite flatbuffer model by
    dumping internal graph structure. It also provides additional features
    like checking GPU delegate compatibility.

    WARNING: Experimental interface, subject to change.
             The output format is not guaranteed to stay stable, so don't
             write scripts to this.

    Args:
      model_path: TFLite flatbuffer model path.
      model_content: TFLite flatbuffer model object.
      gpu_compatibility: Whether to check GPU delegate compatibility.
      **kwargs: Experimental keyword arguments to analyze API.

    Returns:
      Print analyzed report via console output.
    """
    if not model_path and (not model_content):
        raise ValueError('neither `model_path` nor `model_content` is provided')
    if model_path:
        print(f'=== {model_path} ===\n')
        tflite_model = model_path
        input_is_filepath = True
    else:
        print('=== TFLite ModelAnalyzer ===\n')
        tflite_model = model_content
        input_is_filepath = False
    if kwargs.get('experimental_use_mlir', False):
        print(wrap_toco.wrapped_flat_buffer_file_to_mlir(tflite_model, input_is_filepath))
    else:
        print(_analyzer_wrapper.ModelAnalyzer(tflite_model, input_is_filepath, gpu_compatibility))