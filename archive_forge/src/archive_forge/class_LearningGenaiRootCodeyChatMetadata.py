from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootCodeyChatMetadata(_messages.Message):
    """Stores all metadata relating to AIDA DoConversation.

  Enums:
    CodeLanguageValueValuesEnum: Indicates the programming language of the
      code if the message is a code chunk.

  Fields:
    codeLanguage: Indicates the programming language of the code if the
      message is a code chunk.
  """

    class CodeLanguageValueValuesEnum(_messages.Enum):
        """Indicates the programming language of the code if the message is a
    code chunk.

    Values:
      UNSPECIFIED: Unspecified Language.
      ALL: All languages.
      TEXT: Not code.
      CPP: The most common, well-supported languages. C++ code.
      PYTHON: Python code.
      KOTLIN: Kotlin code.
      JAVA: Java code.
      JAVASCRIPT: JavaScript code.
      GO: Go code.
      R: R code.
      JUPYTER_NOTEBOOK: Jupyter notebook.
      TYPESCRIPT: TypeScript code.
      HTML: HTML code.
      SQL: SQL code.
      BASH: Other languages in alphabetical order. BASH code.
      C: C code.
      DART: Dart code.
      GRADLE: Gradle code.
      GROOVY: Groovy code.
      JAVADOC: API documentation.
      JSON: JSON code.
      MAKEFILE: Makefile code.
      MARKDOWN: Markdown code.
      PROTO: Protocol buffer.
      XML: XML code.
      YAML: YAML code.
    """
        UNSPECIFIED = 0
        ALL = 1
        TEXT = 2
        CPP = 3
        PYTHON = 4
        KOTLIN = 5
        JAVA = 6
        JAVASCRIPT = 7
        GO = 8
        R = 9
        JUPYTER_NOTEBOOK = 10
        TYPESCRIPT = 11
        HTML = 12
        SQL = 13
        BASH = 14
        C = 15
        DART = 16
        GRADLE = 17
        GROOVY = 18
        JAVADOC = 19
        JSON = 20
        MAKEFILE = 21
        MARKDOWN = 22
        PROTO = 23
        XML = 24
        YAML = 25
    codeLanguage = _messages.EnumField('CodeLanguageValueValuesEnum', 1)