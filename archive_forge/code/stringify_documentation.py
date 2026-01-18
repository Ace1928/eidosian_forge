import base64
import inspect
import builtins
Create an instance from a JSON style dict.

        Instantiate this class with parameters specified by the dict.

        This method takes the following arguments.

        .. tabularcolumns:: |l|L|

        =============== =====================================================
        Argument        Descrpition
        =============== =====================================================
        dict\_          A dictionary which describes the parameters.
                        For example, {"Param1": 100, "Param2": 200}
        decode_string   (Optional) specify how to decode strings.
                        The default is base64.
                        This argument is used only for attributes which don't
                        have explicit type annotations in _TYPE class
                        attribute.
        additional_args (Optional) Additional kwargs for constructor.
        =============== =====================================================
        