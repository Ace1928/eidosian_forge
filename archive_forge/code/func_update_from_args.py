import json
import urllib.parse as urllib_parse
def update_from_args(self, args):
    if 'justMyCode' in args:
        self.just_my_code = bool_parser(args['justMyCode'])
    elif 'debugStdLib' in args:
        self.just_my_code = not bool_parser(args['debugStdLib'])
    if 'redirectOutput' in args:
        self.redirect_output = bool_parser(args['redirectOutput'])
    if 'showReturnValue' in args:
        self.show_return_value = bool_parser(args['showReturnValue'])
    if 'breakOnSystemExitZero' in args:
        self.break_system_exit_zero = bool_parser(args['breakOnSystemExitZero'])
    if 'django' in args:
        self.django_debug = bool_parser(args['django'])
    if 'flask' in args:
        self.flask_debug = bool_parser(args['flask'])
    if 'jinja' in args:
        self.flask_debug = bool_parser(args['jinja'])
    if 'stopOnEntry' in args:
        self.stop_on_entry = bool_parser(args['stopOnEntry'])
    self.max_exception_stack_frames = int_parser(args.get('maxExceptionStackFrames', 0))
    if 'guiEventLoop' in args:
        self.gui_event_loop = str(args['guiEventLoop'])
    if 'clientOS' in args:
        self.client_os = str(args['clientOS']).upper()