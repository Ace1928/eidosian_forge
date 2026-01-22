import io
import os
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
class HTMLGenerator:
    """Utility class to generate an HTML report."""

    def __init__(self, html_template_path, export_report_path):
        """Reads the HTML template content.

    Args:
      html_template_path: A string, path to the template HTML file.
      export_report_path: A string, path to the generated HTML report. This path
        should point to a '.html' file with date and time in its name.
        e.g. 2019-01-01-10:05.toco_report.html.

    Raises:
      IOError: File doesn't exist.
    """
        if not _file_io.file_exists(html_template_path):
            raise IOError("File '{0}' does not exist.".format(html_template_path))
        with _file_io.FileIO(html_template_path, 'r') as f:
            self.html_template = f.read()
        _file_io.recursive_create_dir(os.path.dirname(export_report_path))
        self.export_report_path = export_report_path

    def generate(self, toco_conversion_log_before, toco_conversion_log_after, post_training_quant_enabled, dot_before, dot_after, toco_err_log='', tflite_graph_path=''):
        """Generates the HTML report and writes it to local directory.

    This function uses the fields in `toco_conversion_log_before` and
    `toco_conversion_log_after` to populate the HTML content. Certain markers
    (placeholders) in the HTML template are then substituted with the fields
    from the protos. Once finished it will write the HTML file to the specified
    local file path.

    Args:
      toco_conversion_log_before: A `TocoConversionLog` protobuf generated
        before the model is converted by TOCO.
      toco_conversion_log_after: A `TocoConversionLog` protobuf generated after
        the model is converted by TOCO.
      post_training_quant_enabled: A boolean, whether post-training quantization
        is enabled.
      dot_before: A string, the dot representation of the model
        before the conversion.
      dot_after: A string, the dot representation of the model after
        the conversion.
      toco_err_log: A string, the logs emitted by TOCO during conversion. Caller
        need to ensure that this string is properly anonymized (any kind of
        user data should be eliminated).
      tflite_graph_path: A string, the filepath to the converted TFLite model.

    Raises:
      RuntimeError: When error occurs while generating the template.
    """
        html_dict = {}
        html_dict['<!--CONVERSION_STATUS-->'] = '<span class="label label-danger">Fail</span>' if toco_err_log else '<span class="label label-success">Success</span>'
        html_dict['<!--TOTAL_OPS_BEFORE_CONVERT-->'] = str(toco_conversion_log_before.model_size)
        html_dict['<!--TOTAL_OPS_AFTER_CONVERT-->'] = str(toco_conversion_log_after.model_size)
        html_dict['<!--BUILT_IN_OPS_COUNT-->'] = str(sum(toco_conversion_log_after.built_in_ops.values()))
        html_dict['<!--SELECT_OPS_COUNT-->'] = str(sum(toco_conversion_log_after.select_ops.values()))
        html_dict['<!--CUSTOM_OPS_COUNT-->'] = str(sum(toco_conversion_log_after.custom_ops.values()))
        html_dict['<!--POST_TRAINING_QUANT_ENABLED-->'] = 'is' if post_training_quant_enabled else "isn't"
        pre_op_profile = ''
        post_op_profile = ''
        for i in range(len(toco_conversion_log_before.op_list)):
            pre_op_profile += '<tr><td>' + toco_conversion_log_before.op_list[i] + '</td>'
            if i < len(toco_conversion_log_before.op_signatures):
                pre_op_profile += '<td>' + get_input_type_from_signature(toco_conversion_log_before.op_signatures[i]) + '</td></tr>'
            else:
                pre_op_profile += '<td></td></tr>'
        for op in toco_conversion_log_after.op_list:
            supported_type = get_operator_type(op, toco_conversion_log_after)
            post_op_profile += '<tr><td>' + op + '</td><td>' + supported_type + '</td></tr>'
        html_dict['<!--REPEAT_TABLE1_ROWS-->'] = pre_op_profile
        html_dict['<!--REPEAT_TABLE2_ROWS-->'] = post_op_profile
        html_dict['<!--DOT_BEFORE_CONVERT-->'] = dot_before
        html_dict['<!--DOT_AFTER_CONVERT-->'] = dot_after
        if toco_err_log:
            html_dict['<!--TOCO_INFO_LOG-->'] = html_escape(toco_err_log)
        else:
            success_info = 'TFLite graph conversion successful. You can preview the converted model at: ' + tflite_graph_path
            html_dict['<!--TOCO_INFO_LOG-->'] = html_escape(success_info)
        template = self.html_template
        for marker in html_dict:
            template = template.replace(marker, html_dict[marker], 1)
            if template.find(marker) != -1:
                raise RuntimeError('Could not populate marker text %r' % marker)
        with _file_io.FileIO(self.export_report_path, 'w') as f:
            f.write(template)