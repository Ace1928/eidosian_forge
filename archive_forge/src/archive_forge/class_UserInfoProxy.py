from __future__ import annotations
from typing import Iterator, Mapping, NoReturn, Union
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import UserInfo
class UserInfoProxy(Mapping[str, Union[str, None]]):
    """
    A read-only, dict-like object for accessing information about current user.

    ``st.experimental_user`` is dependant on the host platform running the
    Streamlit app. If the host platform has not configured the function, it
    will behave as it does in a locally running app.

    Properties can by accessed via key or attribute notation. For example,
    ``st.experimental_user["email"]`` or ``st.experimental_user.email``.

    Parameters
    ----------
    email:str
        If running locally, this property returns the string literal
        ``"test@example.com"``.

        If running on Streamlit Community Cloud, this
        property returns one of two values:

        * ``None`` if the user is not logged in or not a member of the app's        workspace. Such users appear under anonymous pseudonyms in the app's        analytics.
        * The user's email if the the user is logged in and a member of the        app's workspace. Such users are identified by their email in the app's        analytics.

    """

    def __getitem__(self, key: str) -> str | None:
        return _get_user_info()[key]

    def __getattr__(self, key: str) -> str | None:
        try:
            return _get_user_info()[key]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name: str, value: str | None) -> NoReturn:
        raise StreamlitAPIException('st.experimental_user cannot be modified')

    def __setitem__(self, name: str, value: str | None) -> NoReturn:
        raise StreamlitAPIException('st.experimental_user cannot be modified')

    def __iter__(self) -> Iterator[str]:
        return iter(_get_user_info())

    def __len__(self) -> int:
        return len(_get_user_info())

    def to_dict(self) -> UserInfo:
        """
        Get user info as a dictionary.

        This method primarily exists for internal use and is not needed for
        most cases. ``st.experimental_user`` returns an object that inherits from
        ``dict`` by default.

        Returns
        -------
        Dict[str,str]
            A dictionary of the current user's information.
        """
        return _get_user_info()