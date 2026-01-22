import os
from distutils.dist import Distribution
def _get_var(self, name, conf_desc):
    hook, envvar, confvar, convert, append = conf_desc
    if convert is None:
        convert = lambda x: x
    var = self._hook_handler(name, hook)
    if envvar is not None:
        envvar_contents = os.environ.get(envvar)
        if envvar_contents is not None:
            envvar_contents = convert(envvar_contents)
            if var and append:
                if os.environ.get('NPY_DISTUTILS_APPEND_FLAGS', '1') == '1':
                    var.extend(envvar_contents)
                else:
                    var = envvar_contents
            else:
                var = envvar_contents
    if confvar is not None and self._conf:
        if confvar in self._conf:
            source, confvar_contents = self._conf[confvar]
            var = convert(confvar_contents)
    return var