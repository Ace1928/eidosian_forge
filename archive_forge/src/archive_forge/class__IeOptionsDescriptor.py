from enum import Enum
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
class _IeOptionsDescriptor:
    """_IeOptionsDescriptor is an implementation of Descriptor Protocol:

    : Any look-up or assignment to the below attributes in `Options` class will be intercepted
    by `__get__` and `__set__` method respectively.

    - `browser_attach_timeout`
    - `element_scroll_behavior`
    - `ensure_clean_session`
    - `file_upload_dialog_timeout`
    - `force_create_process_api`
    - `force_shell_windows_api`
    - `full_page_screenshot`
    - `ignore_protected_mode_settings`
    - `ignore_zoom_level`
    - `initial_browser_url`
    - `native_events`
    - `persistent_hover`
    - `require_window_focus`
    - `use_per_process_proxy`
    - `use_legacy_file_upload_dialog_handling`
    - `attach_to_edge_chrome`
    - `edge_executable_path`


    : When an attribute lookup happens,
    Example:
        `self. browser_attach_timeout`
        `__get__` method does a dictionary look up in the dictionary `_options` in `Options` class
        and returns the value of key `browserAttachTimeout`
    : When an attribute assignment happens,
    Example:
        `self.browser_attach_timeout` = 30
        `__set__` method sets/updates the value of the key `browserAttachTimeout` in `_options`
        dictionary in `Options` class.
    """

    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, obj, cls):
        return obj._options.get(self.name)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, self.expected_type):
            raise ValueError(f'{self.name} should be of type {self.expected_type.__name__}')
        if self.name == 'elementScrollBehavior' and value not in [ElementScrollBehavior.TOP, ElementScrollBehavior.BOTTOM]:
            raise ValueError('Element Scroll Behavior out of range.')
        obj._options[self.name] = value