from hashlib import md5 as md5_hash
from ..common.by import By
from .command import Command
class ShadowRoot:

    def __init__(self, session, id_) -> None:
        self.session = session
        self._id = id_

    def __eq__(self, other_shadowroot) -> bool:
        return self._id == other_shadowroot._id

    def __hash__(self) -> int:
        return int(md5_hash(self._id.encode('utf-8')).hexdigest(), 16)

    def __repr__(self) -> str:
        return '<{0.__module__}.{0.__name__} (session="{1}", element="{2}")>'.format(type(self), self.session.session_id, self._id)

    def find_element(self, by: str=By.ID, value: str=None):
        if by == By.ID:
            by = By.CSS_SELECTOR
            value = f'[id="{value}"]'
        elif by == By.CLASS_NAME:
            by = By.CSS_SELECTOR
            value = f'.{value}'
        elif by == By.NAME:
            by = By.CSS_SELECTOR
            value = f'[name="{value}"]'
        return self._execute(Command.FIND_ELEMENT_FROM_SHADOW_ROOT, {'using': by, 'value': value})['value']

    def find_elements(self, by: str=By.ID, value: str=None):
        if by == By.ID:
            by = By.CSS_SELECTOR
            value = f'[id="{value}"]'
        elif by == By.CLASS_NAME:
            by = By.CSS_SELECTOR
            value = f'.{value}'
        elif by == By.NAME:
            by = By.CSS_SELECTOR
            value = f'[name="{value}"]'
        return self._execute(Command.FIND_ELEMENTS_FROM_SHADOW_ROOT, {'using': by, 'value': value})['value']

    def _execute(self, command, params=None):
        """Executes a command against the underlying HTML element.

        Args:
          command: The name of the command to _execute as a string.
          params: A dictionary of named parameters to send with the command.

        Returns:
          The command's JSON response loaded into a dictionary object.
        """
        if not params:
            params = {}
        params['shadowId'] = self._id
        return self.session.execute(command, params)