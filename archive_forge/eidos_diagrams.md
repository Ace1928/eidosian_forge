     ```mermaid
     graph TB
         %% Input Layer
         User["User Input"] --> InputProcessor["Input Processor"]
         Config["Configuration System"] --> InputProcessor
         Logger["Logging System"] --> InputProcessor

         %% Knowledge Graph System
         subgraph "Knowledge Graph System"
             RawGraphRAG["Raw Data GraphRAG\n(knowledge/raw)"]
             VerifiedGraphRAG["Verified Facts GraphRAG\n(knowledge/verified)"]
             SpeculationGraphRAG["Speculation Space GraphRAG\n(knowledge/speculative)"]
             IdentityGraphRAG["Identity GraphRAG\n(knowledge/identity)"]
             TimelineGraphRAG["Timeline GraphRAG\n(knowledge/timeline)"]
             DiaryGraphRAG["Diary GraphRAG\n(knowledge/diary)"]
             CitedGraphRAG["Cited GraphRAG\n(knowledge/citeweb)"]
         end

         %% Processing Layer
         subgraph "Processing System"
             TaskAssessor["Task Assessor"]
             ToolMaster["Tool Master"]
             QueryMaster["Query Master"]
             RAGMaster["RAG Master"]
         end

         %% Central LLM and Processing
         EidosLLM["Eidos LLM"] --> |"Orchestrates"| ProcessingSystem

         %% Data Flow
         InputProcessor --> |"Standardized Input"| TaskAssessor
         TaskAssessor --> ToolMaster
         TaskAssessor --> QueryMaster
         TaskAssessor --> RAGMaster

         %% Tool Master Flow
         ToolMaster --> |"Code Execution"| Sandbox["Sandbox Environment"]
         Sandbox --> EidosLLM

         %% Query Master Flow
         QueryMaster --> |"Web Queries"| WebSearch["Web Search & Citation"]
         WebSearch --> CitedGraphRAG
         WebSearch --> EidosLLM

         %% RAG Master Flow
         RAGMaster --> RawGraphRAG
         RAGMaster --> VerifiedGraphRAG
         RAGMaster --> SpeculationGraphRAG

         %% Knowledge Graph Interactions
         RawGraphRAG --> SpeculationGraphRAG
         SpeculationGraphRAG --> VerifiedGraphRAG
         VerifiedGraphRAG --> CitedGraphRAG

         %% Output Processing
         EidosLLM --> RecursivePrompt["Recursive Prompt System"]
         RecursivePrompt --> Critic["Critic & Assessment"]
         Critic --> |"Feedback Loop"| EidosLLM
         Critic --> Formatter["Response Formatter"]
         Formatter --> Output["Final Output"]

         %% Adaptive Control
         subgraph "Adaptive Control"
             Parameters["Parameter Control\n- Token Length\n- Cycle Depth\n- Analysis Depth"]
         end

         EidosLLM --> Parameters
         Parameters --> |"Adjusts"| EidosLLM
         ```
    ```mermaid
    graph TB
    %% Input Layer
    subgraph "Input Processing"
        UserInput["User Input\n- Text, Code, Files\n- API Requests"]
        ConfigSystem["Configuration System\n- LLMConfig, Settings\n- User Preferences"]
        LoggingSystem["Logging System\n- Activity Logs\n- Performance Metrics"]
        StateMonitor["System Monitor\n- Resource Usage\n- Health Checks"]
        UserInput --> InputProcessor["Input Processor\n- Parsing\n- Intent Recognition"]
        ConfigSystem --> InputProcessor
        LoggingSystem --> InputProcessor
        StateMonitor --> InputProcessor
    end
    InputProcessor --> TaskAssessor
    
    %% Knowledge Graph System
    subgraph "Knowledge Graphs"
        RawGraphRAG["Raw Data GraphRAG\n(knowledge/raw)"] --> SpeculationGraphRAG
        SpeculationGraphRAG["Speculation GraphRAG\n(knowledge/speculative)"] --> VerifiedGraphRAG
        VerifiedGraphRAG["Verified Facts GraphRAG\n(knowledge/verified)"] --> CitedGraphRAG
        IdentityGraphRAG["Identity GraphRAG\n(knowledge/identity)"] --> EidosLLM
        TimelineGraphRAG["Timeline GraphRAG\n(knowledge/timeline)"] --> EidosLLM
        DiaryGraphRAG["Diary GraphRAG\n(knowledge/diary)"] --> EidosLLM
        CitedGraphRAG["Cited GraphRAG\n(knowledge/citeweb)"] --> EidosLLM
        LessonsGraph["Lessons GraphRAG\n(knowledge/lessons)"] --> EidosLLM
        MajorMomentsGraph["Major Moments\n(knowledge/majormoments)"] --> EidosLLM
        ProcessedGraph["Processed Data\n(knowledge/processed)"] --> EidosLLM
        RefinedGraph["Refined Data\n(knowledge/refined)"] --> EidosLLM
    end
    
    %% Core Processing
    subgraph "Processing Core"
        EidosLLM["Eidos LLM\n- Core Reasoning\n- Orchestration"] --> TaskAssessor["Task Assessor\n- Complexity Analysis"]
        TaskAssessor --> ToolMaster["Tool Master\n- Code Execution"]
        TaskAssessor --> QueryMaster["Query Master\n- Information Retrieval"]
        TaskAssessor --> RAGMaster["RAG Master\n- Knowledge Access"]
        TaskAssessor --> SpecializedModules["Specialized Modules\n- Translation\n- Style Transfer"]
    end
    
    %% Processing Flows
    ToolMaster --> Sandbox["Secure Sandbox"] --> EidosLLM
    QueryMaster --> WebSearch["Web Search"] --> CitedGraphRAG
    WebSearch --> EidosLLM
    RAGMaster --> RawGraphRAG
    RAGMaster --> VerifiedGraphRAG
    RAGMaster --> SpeculationGraphRAG
    SpecializedModules --> EidosLLM
    
    %% Feedback System
    subgraph "Evaluation"
        InternalCritic["Internal Critic"] --> DualCritic
        ExternalCritic["External Critic"] --> DualCritic
        RecursivePrompt["Recursive Prompting"] --> DualCritic["Dual-Critic System"]
    end
    
    %% Adaptive Control
    subgraph "Adaptive Systems"
        ResourceMonitor["Resource Monitor"] --> Parameters["Parameter Control"]
        PerformanceTracker["Performance Tracker"] --> Parameters
    end
    
    %% Final Flow
    EidosLLM --> RecursivePrompt
    DualCritic --> Formatter["Response Formatter"]
    Formatter --> Output["Final Output"]
    Parameters --> EidosLLM
    DualCritic --> EidosLLM
    ```